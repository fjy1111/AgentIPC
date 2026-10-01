from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agentipc.artifacts.store import ArtifactStore
from agentipc.evaluation.text_counter import TextCounter
from agentipc.protocol.codec import encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageType
from agentipc.protocol.text_adapter import render


DEFAULT_PAYLOAD_SIZES = (1024, 8192, 32768, 131072)


def run_e5(
    *,
    root: Path,
    repeat: int,
    payload_sizes: tuple[int, ...] = DEFAULT_PAYLOAD_SIZES,
    text_counter: TextCounter | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be >= 1")
    if (
        type(payload_sizes) is not tuple
        or not payload_sizes
        or any(type(size) is not int or size <= 0 for size in payload_sizes)
    ):
        raise ValueError("payload_sizes must be a non-empty tuple of positive ints")

    counter = TextCounter() if text_counter is None else text_counter
    if not isinstance(counter, TextCounter):
        raise TypeError("text_counter must be a TextCounter or None")
    probe = counter.count("AgentIPC E5 tokenizer probe.")
    if probe.token_method == "unavailable":
        raise RuntimeError(
            "E5 requires tiktoken for comparable wire-token accounting; "
            "install with: pip install -e '.[tiktoken]'"
        )

    root.mkdir(parents=True, exist_ok=True)
    artifact_store = ArtifactStore(root / "artifacts")
    rows: list[dict[str, Any]] = []

    for size in payload_sizes:
        payload_text = _build_payload(size)
        assert len(payload_text.encode("utf-8")) == size
        for repeat_index in range(repeat):
            common = {
                "version": "1.0",
                "message_id": f"e5-{size}-{repeat_index}",
                "trace_id": f"e5-trace-{size}-{repeat_index}",
                "task_id": f"e5-task-{size}-{repeat_index}",
                "step_id": "step-transfer",
                "sender": "producer",
                "receiver": "consumer",
                "message_type": MessageType.REQUEST,
                "action": ActionType.EXECUTE,
                "created_at": 0.0,
            }

            materialized = AgentEnvelope(
                **common,
                args={
                    "payload_kind": "materialized_text",
                    "payload": payload_text,
                },
            )
            rows.append(_measure_text(materialized, counter, size, repeat_index))
            rows.append(
                _measure_structured(
                    "B", materialized, counter, size, repeat_index, 0
                )
            )

            artifact_ref = artifact_store.put_bytes(
                payload_text.encode("utf-8"),
                media_type="text/plain; charset=utf-8",
                summary=f"E5 deterministic {size}-byte payload",
            )
            referenced = AgentEnvelope(
                **common,
                args={
                    "payload_kind": "artifact_ref",
                    "payload_size_bytes": size,
                },
                artifact_refs=[artifact_ref],
            )
            rows.append(
                _measure_structured(
                    "C", referenced, counter, size, repeat_index, size
                )
            )

    return rows, _aggregate(rows, repeat=repeat, token_method=probe.token_method)


def _build_payload(size: int) -> str:
    chunks: list[str] = []
    index = 0
    total = 0
    while total < size:
        line = (
            f"record={index:06d};agent=retriever;topic=openeuler;"
            "context=deterministic communication payload;status=ok\n"
        )
        chunks.append(line)
        total += len(line.encode("ascii"))
        index += 1
    return "".join(chunks).encode("ascii")[:size].decode("ascii")


def _measure_text(
    envelope: AgentEnvelope,
    counter: TextCounter,
    size: int,
    repeat_index: int,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    wire_text = render(envelope)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    count = counter.count(wire_text)
    return {
        "experiment": "A",
        "payload_bytes": size,
        "repeat": repeat_index,
        "message_count": 1,
        "wire_chars": count.text_chars,
        "wire_tokens": count.text_tokens,
        "wire_bytes": len(wire_text.encode("utf-8")),
        "text_chars": count.text_chars,
        "text_tokens": count.text_tokens,
        "protocol_bytes": 0,
        "state_transfer_count": 0,
        "state_bytes": 0,
        "artifact_ref_count": 0,
        "artifact_payload_bytes": 0,
        "latency_ms": latency_ms,
        "token_method": count.token_method,
    }


def _measure_structured(
    mode: str,
    envelope: AgentEnvelope,
    counter: TextCounter,
    size: int,
    repeat_index: int,
    artifact_payload_bytes: int,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    payload = encode(envelope)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    count = counter.count(payload.decode("utf-8"))
    return {
        "experiment": mode,
        "payload_bytes": size,
        "repeat": repeat_index,
        "message_count": 1,
        "wire_chars": count.text_chars,
        "wire_tokens": count.text_tokens,
        "wire_bytes": len(payload),
        "text_chars": 0,
        "text_tokens": 0,
        "protocol_bytes": len(payload),
        "state_transfer_count": 0,
        "state_bytes": 0,
        "artifact_ref_count": len(envelope.artifact_refs),
        "artifact_payload_bytes": artifact_payload_bytes,
        "latency_ms": latency_ms,
        "token_method": count.token_method,
    }


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    repeat: int,
    token_method: str,
) -> dict[str, Any]:
    payloads: dict[str, Any] = {}
    for size in sorted({int(row["payload_bytes"]) for row in rows}):
        size_rows = [row for row in rows if row["payload_bytes"] == size]
        modes: dict[str, Any] = {}
        for mode in ("A", "B", "C"):
            selected = [row for row in size_rows if row["experiment"] == mode]
            modes[mode] = {
                "mean_wire_chars": _mean(selected, "wire_chars"),
                "mean_wire_tokens": _mean(selected, "wire_tokens"),
                "mean_wire_bytes": _mean(selected, "wire_bytes"),
                "mean_latency_ms": _mean(selected, "latency_ms"),
                "mean_state_bytes": _mean(selected, "state_bytes"),
                "mean_artifact_payload_bytes": _mean(selected, "artifact_payload_bytes"),
            }
        baseline = modes["A"]
        savings = {}
        for mode in ("B", "C"):
            current = modes[mode]
            savings[mode] = {
                "wire_token_saving_pct": _saving_pct(
                    baseline["mean_wire_tokens"], current["mean_wire_tokens"]
                ),
                "wire_byte_saving_pct": _saving_pct(
                    baseline["mean_wire_bytes"], current["mean_wire_bytes"]
                ),
                "total_transfer_saving_pct": _saving_pct(
                    baseline["mean_wire_bytes"],
                    current["mean_wire_bytes"] + current["mean_state_bytes"],
                ),
            }
        payloads[str(size)] = {
            "payload_bytes": size,
            "modes": modes,
            "savings_vs_A": savings,
        }
    return {
        "experiment": "E5",
        "repeat": repeat,
        "token_method": token_method,
        "payloads": payloads,
        "note": (
            "artifact_payload_bytes is disclosed separately and is not counted "
            "as agent-to-agent wire bytes; C sends an ArtifactRef on the wire."
        ),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _saving_pct(baseline: float, current: float) -> float:
    if baseline <= 0.0:
        raise ValueError("baseline must be positive for a saving percentage")
    return (1.0 - current / baseline) * 100.0
