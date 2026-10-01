from __future__ import annotations

import multiprocessing as mp
import statistics
import time
from multiprocessing.connection import Connection
from typing import Any, Iterable

import numpy as np

from agentipc.evaluation.text_counter import TextCounter
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import MessageType
from agentipc.state.checksum import array_checksum
from agentipc.state.hub import StateHub


E7_VECTOR_DIMS: tuple[int, ...] = (64, 256, 1024, 4096, 16384, 65536)
E7_DTYPE = np.dtype(np.float32)
E7_WORKER_TIMEOUT_SEC = 30.0
_STOP = b"__agentipc_e7_stop__"


def run_e7(
    *,
    repeat: int,
    vector_dims: Iterable[int] = E7_VECTOR_DIMS,
    use_tiktoken: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Benchmark materialized JSON state vs SharedMemory + StateRef.

    The consumer is a separate spawned process. Process startup is completed
    before any timed sample. The downstream consumer operation is a float64
    sum over the reconstructed float32 state.

    ``wire_bytes`` measures the encoded AgentEnvelope bytes sent through the
    multiprocessing Pipe and excludes OS Pipe framing. Shared-memory payload
    bytes are disclosed separately through ``state_bytes``.
    """
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be an int >= 1")

    dims = _validate_vector_dims(vector_dims)
    counter = TextCounter(use_tiktoken=use_tiktoken)

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    process = ctx.Process(
        target=_consumer_worker,
        args=(child_conn,),
        name="agentipc-e7-consumer",
    )
    process.start()
    child_conn.close()

    rows: list[dict[str, Any]] = []
    try:
        _wait_worker_ready(parent_conn, process)

        for dim in dims:
            array = _build_state_vector(dim)
            expected_checksum = array_checksum(array)
            expected_value = _consumer_operation(array)

            for repeat_index in range(repeat):
                # Alternate order to reduce systematic warm-cache/order bias.
                mode_order = (
                    ("json_materialized", "shm_ref")
                    if repeat_index % 2 == 0
                    else ("shm_ref", "json_materialized")
                )
                for mode in mode_order:
                    if mode == "json_materialized":
                        row = _exchange_json(
                            conn=parent_conn,
                            process=process,
                            array=array,
                            expected_checksum=expected_checksum,
                            expected_value=expected_value,
                            repeat_index=repeat_index,
                            counter=counter,
                        )
                    else:
                        row = _exchange_shm(
                            conn=parent_conn,
                            process=process,
                            array=array,
                            expected_checksum=expected_checksum,
                            expected_value=expected_value,
                            repeat_index=repeat_index,
                            counter=counter,
                        )
                    rows.append(row)
    finally:
        _stop_worker(parent_conn, process)

    token_methods = {str(row["token_method"]) for row in rows}
    if use_tiktoken and token_methods == {"unavailable"}:
        raise RuntimeError(
            "E7 requires tiktoken for formal wire-token accounting; "
            'install with: python -m pip install -e ".[tiktoken]"'
        )

    return rows, _aggregate_e7(rows, repeat=repeat)


def _validate_vector_dims(vector_dims: Iterable[int]) -> tuple[int, ...]:
    try:
        dims = tuple(vector_dims)
    except TypeError as exc:
        raise TypeError("vector_dims must be an iterable of positive ints") from exc

    if not dims:
        raise ValueError("vector_dims must not be empty")

    normalized: list[int] = []
    seen: set[int] = set()
    for dim in dims:
        if type(dim) is not int or dim < 1:
            raise ValueError("vector_dims must contain only positive ints")
        if dim in seen:
            raise ValueError("vector_dims must not contain duplicates")
        seen.add(dim)
        normalized.append(dim)
    return tuple(normalized)


def _build_state_vector(dim: int) -> np.ndarray:
    # Deterministic synthetic float32 state. 1024 dimensions matches the
    # embedding width used by the current real-provider experiments.
    values = np.arange(dim, dtype=np.float32)
    values /= np.float32(max(dim, 1))
    return np.ascontiguousarray(values, dtype=np.float32)


def _consumer_operation(array: np.ndarray) -> float:
    return float(np.sum(array, dtype=np.float64))


def _base_envelope(
    *,
    args: dict[str, Any],
    state_refs: list[Any],
) -> AgentEnvelope:
    return AgentEnvelope(
        message_id="e7-message",
        trace_id="e7-trace",
        task_id="e7-task",
        step_id="e7-state-exchange",
        sender="producer",
        receiver="consumer",
        message_type=MessageType.REQUEST,
        capability="state_exchange",
        args=args,
        state_refs=state_refs,
        artifact_refs=[],
        memory_refs=[],
        created_at=0.0,
    )


def _exchange_json(
    *,
    conn: Connection,
    process: mp.Process,
    array: np.ndarray,
    expected_checksum: str,
    expected_value: float,
    repeat_index: int,
    counter: TextCounter,
) -> dict[str, Any]:
    e2e_started = time.perf_counter_ns()
    producer_started = time.perf_counter_ns()

    envelope = _base_envelope(
        args={
            "mode": "json_materialized",
            "shape": list(array.shape),
            "dtype": array.dtype.str,
            "state": array.tolist(),
        },
        state_refs=[],
    )
    payload = encode(envelope)
    producer_ms = _elapsed_ms(producer_started)

    send_started = time.perf_counter_ns()
    conn.send_bytes(payload)
    send_ms = _elapsed_ms(send_started)

    completion = _recv_phase(conn, process, expected_phase="complete")
    end_to_end_ms = _elapsed_ms(e2e_started)
    validation = _recv_phase(conn, process, expected_phase="validation")

    return _build_row(
        mode="json_materialized",
        array=array,
        repeat_index=repeat_index,
        payload=payload,
        producer_ms=producer_ms,
        send_ms=send_ms,
        end_to_end_ms=end_to_end_ms,
        completion=completion,
        validation=validation,
        expected_checksum=expected_checksum,
        expected_value=expected_value,
        state_transfer_count=0,
        state_bytes=0,
        counter=counter,
    )


def _exchange_shm(
    *,
    conn: Connection,
    process: mp.Process,
    array: np.ndarray,
    expected_checksum: str,
    expected_value: float,
    repeat_index: int,
    counter: TextCounter,
) -> dict[str, Any]:
    hub = StateHub(transport="shm")
    ref = None
    try:
        e2e_started = time.perf_counter_ns()
        producer_started = time.perf_counter_ns()

        ref = hub.put_array(
            array,
            kind="formal_e7_state",
            summary="cross-process non-text state exchange",
        )
        envelope = _base_envelope(
            args={"mode": "shm_ref"},
            state_refs=[ref],
        )
        payload = encode(envelope)
        producer_ms = _elapsed_ms(producer_started)

        send_started = time.perf_counter_ns()
        conn.send_bytes(payload)
        send_ms = _elapsed_ms(send_started)

        completion = _recv_phase(conn, process, expected_phase="complete")
        end_to_end_ms = _elapsed_ms(e2e_started)
        validation = _recv_phase(conn, process, expected_phase="validation")

        return _build_row(
            mode="shm_ref",
            array=array,
            repeat_index=repeat_index,
            payload=payload,
            producer_ms=producer_ms,
            send_ms=send_ms,
            end_to_end_ms=end_to_end_ms,
            completion=completion,
            validation=validation,
            expected_checksum=expected_checksum,
            expected_value=expected_value,
            state_transfer_count=1,
            state_bytes=int(ref.nbytes),
            counter=counter,
        )
    finally:
        if ref is not None:
            hub.release(ref)
        hub.close()


def _build_row(
    *,
    mode: str,
    array: np.ndarray,
    repeat_index: int,
    payload: bytes,
    producer_ms: float,
    send_ms: float,
    end_to_end_ms: float,
    completion: dict[str, Any],
    validation: dict[str, Any],
    expected_checksum: str,
    expected_value: float,
    state_transfer_count: int,
    state_bytes: int,
    counter: TextCounter,
) -> dict[str, Any]:
    text = payload.decode("utf-8")
    count = counter.count(text)
    consumer_value = float(completion["consumer_value"])
    checksum_match = validation["checksum"] == expected_checksum
    consumer_result_match = bool(
        np.isclose(consumer_value, expected_value, rtol=1e-12, atol=1e-12)
    )
    consumer_ms = float(completion["consumer_ms"])
    validation_ms = float(validation["validation_ms"])

    return {
        "experiment": "E7",
        "mode": mode,
        "repeat": repeat_index + 1,
        "vector_dim": int(array.size),
        "dtype": array.dtype.str,
        "logical_state_bytes": int(array.nbytes),
        "wire_chars": count.text_chars,
        "wire_tokens": count.text_tokens,
        "wire_bytes": len(payload),
        "token_method": count.token_method,
        "state_transfer_count": state_transfer_count,
        "state_bytes": state_bytes,
        "producer_ms": producer_ms,
        "pipe_send_ms": send_ms,
        "consumer_ms": consumer_ms,
        "validation_ms": validation_ms,
        "end_to_end_ms": end_to_end_ms,
        "ipc_and_sync_ms": max(0.0, end_to_end_ms - producer_ms - consumer_ms),
        "checksum_match": bool(checksum_match),
        "consumer_result_match": consumer_result_match,
        "consumer_value": consumer_value,
        "success": bool(checksum_match and consumer_result_match),
    }


def _consumer_worker(conn: Connection) -> None:
    state_hub = StateHub(transport="shm")
    try:
        conn.send({"ready": True})
        while True:
            payload = conn.recv_bytes()
            if payload == _STOP:
                return

            try:
                consumer_started = time.perf_counter_ns()
                envelope = decode(payload)
                mode = envelope.args.get("mode")

                if mode == "json_materialized":
                    shape = envelope.args.get("shape")
                    dtype_value = envelope.args.get("dtype")
                    state = envelope.args.get("state")
                    if (
                        not isinstance(shape, list)
                        or not isinstance(dtype_value, str)
                        or not isinstance(state, list)
                    ):
                        raise ValueError(
                            "json_materialized envelope is missing state metadata"
                        )
                    array = np.asarray(state, dtype=np.dtype(dtype_value))
                    array = np.ascontiguousarray(array.reshape(tuple(shape)))
                elif mode == "shm_ref":
                    if len(envelope.state_refs) != 1:
                        raise ValueError(
                            "shm_ref envelope requires exactly one StateRef"
                        )
                    array = state_hub.resolve_array(envelope.state_refs[0])
                else:
                    raise ValueError(f"unsupported E7 mode: {mode!r}")

                consumer_value = _consumer_operation(array)
                consumer_ms = _elapsed_ms(consumer_started)

                # This acknowledgement closes the measured critical path.
                conn.send(
                    {
                        "ok": True,
                        "phase": "complete",
                        "consumer_ms": consumer_ms,
                        "consumer_value": consumer_value,
                        "resolved_nbytes": int(array.nbytes),
                    }
                )

                # Exact integrity validation runs after the measured critical path.
                # StateHub has already validated shm_ref inside resolve_array().
                validation_started = time.perf_counter_ns()
                checksum = array_checksum(array)
                validation_ms = _elapsed_ms(validation_started)
                conn.send(
                    {
                        "ok": True,
                        "phase": "validation",
                        "checksum": checksum,
                        "validation_ms": validation_ms,
                    }
                )
            except Exception as exc:
                conn.send(
                    {
                        "ok": False,
                        "phase": "error",
                        "error_type": exc.__class__.__name__,
                        "message": str(exc),
                    }
                )
    finally:
        state_hub.close()
        conn.close()


def _wait_worker_ready(conn: Connection, process: mp.Process) -> None:
    if not conn.poll(E7_WORKER_TIMEOUT_SEC):
        _raise_worker_timeout(process, phase="startup")
    message = conn.recv()
    if not isinstance(message, dict) or message.get("ready") is not True:
        raise RuntimeError(f"E7 consumer returned invalid startup message: {message!r}")


def _recv_phase(
    conn: Connection,
    process: mp.Process,
    *,
    expected_phase: str,
) -> dict[str, Any]:
    if not conn.poll(E7_WORKER_TIMEOUT_SEC):
        _raise_worker_timeout(process, phase=expected_phase)

    message = conn.recv()
    if not isinstance(message, dict):
        raise RuntimeError(f"E7 consumer returned non-dict message: {message!r}")
    if message.get("ok") is not True:
        raise RuntimeError(
            "E7 consumer failed: "
            f"{message.get('error_type', 'UnknownError')}: "
            f"{message.get('message', '')}"
        )
    if message.get("phase") != expected_phase:
        raise RuntimeError(
            f"E7 consumer phase mismatch: expected {expected_phase!r}, "
            f"got {message.get('phase')!r}"
        )
    return message


def _raise_worker_timeout(process: mp.Process, *, phase: str) -> None:
    if not process.is_alive():
        raise RuntimeError(
            f"E7 consumer exited unexpectedly during {phase} "
            f"with exit code {process.exitcode}"
        )
    raise TimeoutError(
        f"E7 consumer did not respond during {phase} within "
        f"{E7_WORKER_TIMEOUT_SEC:.0f}s"
    )


def _stop_worker(conn: Connection, process: mp.Process) -> None:
    try:
        if process.is_alive():
            try:
                conn.send_bytes(_STOP)
            except (BrokenPipeError, EOFError, OSError):
                pass
            process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
    finally:
        conn.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)


def _elapsed_ms(started_ns: int) -> float:
    return (time.perf_counter_ns() - started_ns) / 1_000_000.0


def _aggregate_e7(
    rows: list[dict[str, Any]],
    *,
    repeat: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty E7 row set")

    payloads: dict[str, Any] = {}
    for state_bytes in sorted({int(row["logical_state_bytes"]) for row in rows}):
        selected = [
            row
            for row in rows
            if int(row["logical_state_bytes"]) == state_bytes
        ]
        modes = {
            mode: _aggregate_mode(
                [row for row in selected if row["mode"] == mode]
            )
            for mode in ("json_materialized", "shm_ref")
        }
        baseline = modes["json_materialized"]
        shm = modes["shm_ref"]
        payloads[str(state_bytes)] = {
            "vector_dim": int(selected[0]["vector_dim"]),
            "state_bytes": state_bytes,
            "modes": modes,
            "savings": {
                "wire_token_saving_pct": _saving_pct(
                    baseline["mean_wire_tokens"], shm["mean_wire_tokens"]
                ),
                "wire_byte_saving_pct": _saving_pct(
                    baseline["mean_wire_bytes"], shm["mean_wire_bytes"]
                ),
                "end_to_end_latency_reduction_pct": _saving_pct(
                    baseline["mean_end_to_end_ms"], shm["mean_end_to_end_ms"]
                ),
                "producer_latency_reduction_pct": _saving_pct(
                    baseline["mean_producer_ms"], shm["mean_producer_ms"]
                ),
                "consumer_latency_reduction_pct": _saving_pct(
                    baseline["mean_consumer_ms"], shm["mean_consumer_ms"]
                ),
            },
        }

    token_methods = sorted({str(row["token_method"]) for row in rows})
    expected_rows = len(payloads) * repeat * 2
    correctness_pass = all(bool(row["success"]) for row in rows)
    accounting_pass = all(
        (
            row["state_transfer_count"] == 0
            and row["state_bytes"] == 0
        )
        if row["mode"] == "json_materialized"
        else (
            row["state_transfer_count"] == 1
            and row["state_bytes"] == row["logical_state_bytes"]
        )
        for row in rows
    )

    return {
        "experiment": "E7",
        "repeat": repeat,
        "row_count": len(rows),
        "expected_row_count": expected_rows,
        "passed": bool(
            correctness_pass and accounting_pass and len(rows) == expected_rows
        ),
        "correctness_pass": correctness_pass,
        "accounting_pass": accounting_pass,
        "dtype": E7_DTYPE.str,
        "token_method": ", ".join(token_methods),
        "process_model": "spawned consumer process",
        "ipc_transport": "multiprocessing.Pipe",
        "consumer_operation": "numpy.sum(state, dtype=float64)",
        "timing_scope": (
            "producer state conversion/write + AgentEnvelope encode + Pipe send/receive "
            "+ consumer decode/resolve + downstream consumer operation; process startup "
            "and post-run cleanup are excluded"
        ),
        "wire_scope": (
            "encoded AgentEnvelope bytes sent through Pipe; OS Pipe framing is excluded. "
            "For shm_ref, state_bytes is disclosed separately and the state payload remains "
            "in Linux shared memory."
        ),
        "zero_copy_claim": False,
        "note": (
            "AgentIPC does not claim zero-copy here: StateHub.resolve_array() returns an "
            "owned ndarray copy. E7 measures avoidance of text/JSON materialization for "
            "numeric intermediate state by using SharedMemory + StateRef. Exact checksum "
            "validation used for benchmark correctness is performed after the measured "
            "critical-path acknowledgement; StateHub's own checksum verification remains "
            "inside the shm_ref consumer path."
        ),
        "payloads": payloads,
    }


def _aggregate_mode(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty E7 mode")

    def values(name: str) -> list[float]:
        return [float(row[name]) for row in rows]

    e2e = values("end_to_end_ms")
    return {
        "sample_count": len(rows),
        "success_rate": sum(bool(row["success"]) for row in rows) / len(rows),
        "mean_wire_chars": statistics.mean(values("wire_chars")),
        "mean_wire_tokens": statistics.mean(values("wire_tokens")),
        "mean_wire_bytes": statistics.mean(values("wire_bytes")),
        "mean_state_transfer_count": statistics.mean(
            values("state_transfer_count")
        ),
        "mean_state_bytes": statistics.mean(values("state_bytes")),
        "mean_producer_ms": statistics.mean(values("producer_ms")),
        "mean_pipe_send_ms": statistics.mean(values("pipe_send_ms")),
        "mean_consumer_ms": statistics.mean(values("consumer_ms")),
        "mean_ipc_and_sync_ms": statistics.mean(values("ipc_and_sync_ms")),
        "mean_end_to_end_ms": statistics.mean(e2e),
        "median_end_to_end_ms": statistics.median(e2e),
        "p95_end_to_end_ms": _percentile(e2e, 0.95),
        "mean_validation_ms": statistics.mean(values("validation_ms")),
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _saving_pct(baseline: float, current: float) -> float:
    if baseline <= 0:
        return 0.0
    return (1.0 - (current / baseline)) * 100.0
