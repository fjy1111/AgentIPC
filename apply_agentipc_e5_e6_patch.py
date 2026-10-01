#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def fail(message: str) -> None:
    raise SystemExit(f"[agentipc-e5-e6] {message}")


def read(rel: str) -> str:
    path = ROOT / rel
    if not path.is_file():
        fail(f"missing expected file: {rel}")
    return path.read_text(encoding="utf-8")


def write(rel: str, content: str) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"WRITE {rel}")


def replace_once(rel: str, old: str, new: str) -> None:
    text = read(rel)
    count = text.count(old)
    if count != 1:
        fail(f"{rel}: expected replacement block exactly once, found {count}")
    write(rel, text.replace(old, new, 1))


def ensure_absent(rel: str) -> None:
    if (ROOT / rel).exists():
        fail(f"refusing to overwrite existing new file: {rel}")


def main() -> None:
    if not (ROOT / "pyproject.toml").is_file() or not (ROOT / "src/agentipc").is_dir():
        fail("run this script from the AgentIPC repository root")

    replace_once(
        "src/agentipc/evaluation/metrics.py",
        '''    text_chars: StrictNonNegativeInt = 0
    text_tokens: StrictNonNegativeInt = 0
    protocol_bytes: StrictNonNegativeInt = 0
''',
        '''    text_chars: StrictNonNegativeInt = 0
    text_tokens: StrictNonNegativeInt = 0
    wire_chars: StrictNonNegativeInt = 0
    wire_tokens: StrictNonNegativeInt = 0
    wire_bytes: StrictNonNegativeInt = 0
    protocol_bytes: StrictNonNegativeInt = 0
''',
    )
    replace_once(
        "src/agentipc/evaluation/metrics.py",
        '''    artifact_ref_count: StrictNonNegativeInt = 0

    memory_retrieved: StrictNonNegativeInt = 0
''',
        '''    artifact_ref_count: StrictNonNegativeInt = 0
    artifact_payload_bytes: StrictNonNegativeInt = 0

    memory_retrieved: StrictNonNegativeInt = 0
''',
    )
    replace_once(
        "src/agentipc/evaluation/metrics.py",
        '''    memory_effective: StrictNonNegativeInt = 0
    memory_harmful: StrictNonNegativeInt = 0

    tool_call_count: StrictNonNegativeInt = 0
''',
        '''    memory_effective: StrictNonNegativeInt = 0
    memory_harmful: StrictNonNegativeInt = 0
    fast_path_hit_count: StrictNonNegativeInt = 0

    tool_call_count: StrictNonNegativeInt = 0
''',
    )
    replace_once(
        "src/agentipc/evaluation/metrics.py",
        '''            "text_chars",
            "text_tokens",
            "protocol_bytes",
''',
        '''            "text_chars",
            "text_tokens",
            "wire_chars",
            "wire_tokens",
            "wire_bytes",
            "protocol_bytes",
''',
    )
    replace_once(
        "src/agentipc/evaluation/metrics.py",
        '''            "artifact_ref_count",
            "memory_retrieved",
''',
        '''            "artifact_ref_count",
            "artifact_payload_bytes",
            "memory_retrieved",
''',
    )
    replace_once(
        "src/agentipc/evaluation/metrics.py",
        '''            "memory_effective",
            "memory_harmful",
            "tool_call_count",
''',
        '''            "memory_effective",
            "memory_harmful",
            "fast_path_hit_count",
            "tool_call_count",
''',
    )

    replace_once(
        "src/agentipc/runtime/text_transport.py",
        '''        if isinstance(ctx.metrics, MetricsCollector):
            count = self._text_counter.count(rendered)
            ctx.metrics.increment("message_count")
            ctx.metrics.increment("text_chars", count.text_chars)
            ctx.metrics.increment("text_tokens", count.text_tokens)

        ctx.trace_logger.log_envelope(envelope)
''',
        '''        if isinstance(ctx.metrics, MetricsCollector):
            count = self._text_counter.count(rendered)
            payload = rendered.encode("utf-8")
            ctx.metrics.increment("message_count")
            ctx.metrics.increment("text_chars", count.text_chars)
            ctx.metrics.increment("text_tokens", count.text_tokens)
            ctx.metrics.increment("wire_chars", count.text_chars)
            ctx.metrics.increment("wire_tokens", count.text_tokens)
            ctx.metrics.increment("wire_bytes", len(payload))
            if envelope.state_refs:
                ctx.metrics.increment("state_transfer_count", len(envelope.state_refs))
                ctx.metrics.increment(
                    "state_bytes",
                    sum(ref.nbytes for ref in envelope.state_refs),
                )
            if envelope.artifact_refs:
                ctx.metrics.increment("artifact_ref_count", len(envelope.artifact_refs))
                ctx.metrics.increment(
                    "artifact_payload_bytes",
                    sum(ref.size_bytes for ref in envelope.artifact_refs),
                )

        ctx.trace_logger.log_envelope(envelope)
''',
    )

    replace_once(
        "src/agentipc/memory/sqlite_store.py",
        '''_RECORD_USE_SQL = """
UPDATE memories
SET
''',
        '''_UPDATE_PAYLOAD_SQL = """
UPDATE memories
SET payload_json = ?
WHERE memory_id = ?
"""

_RECORD_USE_SQL = """
UPDATE memories
SET
''',
    )
    replace_once(
        "src/agentipc/memory/sqlite_store.py",
        '''    def record_use(
        self,
        memory_id: str,
''',
        '''    def update_payload(
        self,
        memory_id: str,
        payload: dict[str, Any],
    ) -> MemoryRecord:
        connection = self._ensure_open()
        validated_id = _validate_memory_id(memory_id)
        if type(payload) is not dict:
            raise TypeError("payload must be a dict[str, Any]")

        payload_json = _json_dumps(payload)
        with connection:
            cursor = connection.execute(
                _UPDATE_PAYLOAD_SQL,
                (payload_json, validated_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(validated_id)
            row = connection.execute(
                _SELECT_BY_ID_SQL,
                (validated_id,),
            ).fetchone()
            if row is None:  # pragma: no cover
                raise KeyError(validated_id)
            return _row_to_record(row)

    def record_use(
        self,
        memory_id: str,
''',
    )

    replace_once(
        "src/agentipc/memory/service.py",
        '''from agentipc.providers.base import EmbeddingProvider


class MemoryService:
''',
        '''from agentipc.providers.base import EmbeddingProvider


_VALIDATION_PAYLOAD_KEY = "_agentipc_validation"


class MemoryService:
''',
    )
    replace_once(
        "src/agentipc/memory/service.py",
        '''    def get(self, memory_id: str) -> MemoryRecord | None:
        return self.store.get(memory_id)

    def retrieve(
''',
        '''    def get(self, memory_id: str) -> MemoryRecord | None:
        return self.store.get(memory_id)

    def mark_validated_result(
        self,
        memory_id: str,
        *,
        passed: bool,
    ) -> MemoryRecord:
        if not isinstance(memory_id, str):
            raise TypeError("memory_id must be a str")
        if memory_id == "":
            raise ValueError("memory_id must be non-empty")
        if type(passed) is not bool:
            raise TypeError("passed must be a bool")

        record = self.store.get(memory_id)
        if record is None:
            raise KeyError(memory_id)
        if record.memory_type is not MemoryType.RESULT:
            raise ValueError("only RESULT memory can be evaluator-validated")

        payload = dict(record.payload)
        payload[_VALIDATION_PAYLOAD_KEY] = {
            "passed": passed,
            "source": "external_evaluator",
        }
        return self.store.update_payload(memory_id, payload)

    def get_exact_validated_result(self, task: str) -> MemoryRecord | None:
        if not isinstance(task, str):
            raise TypeError("task must be a str")
        if task == "":
            raise ValueError("task must be non-empty")

        for record in reversed(self.store.list_records()):
            if record.memory_type is not MemoryType.RESULT:
                continue
            if record.task_topic != task:
                continue
            validation = record.payload.get(_VALIDATION_PAYLOAD_KEY)
            if (
                type(validation) is dict
                and validation.get("passed") is True
                and validation.get("source") == "external_evaluator"
            ):
                return record
        return None

    def retrieve(
''',
    )

    replace_once(
        "src/agentipc/runtime/orchestrator.py",
        '''from agentipc.evaluation.metrics import MetricsCollector
''',
        '''from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.text_counter import TextCounter
''',
    )
    replace_once(
        "src/agentipc/runtime/orchestrator.py",
        '''    def __init__(
        self,
        router: Router,
        *,
        text_transport: TextTransport | None = None,
    ) -> None:
''',
        '''    def __init__(
        self,
        router: Router,
        *,
        text_transport: TextTransport | None = None,
        wire_counter: TextCounter | None = None,
        memory_fast_path: bool = False,
    ) -> None:
''',
    )
    replace_once(
        "src/agentipc/runtime/orchestrator.py",
        '''        self._router = router
        self._text_transport = text_transport

    def run_task(
''',
        '''        if wire_counter is None:
            wire_counter = TextCounter()
        elif not isinstance(wire_counter, TextCounter):
            raise TypeError("wire_counter must be a TextCounter or None")
        if type(memory_fast_path) is not bool:
            raise TypeError("memory_fast_path must be a bool")

        self._router = router
        self._text_transport = text_transport
        self._wire_counter = wire_counter
        self._memory_fast_path = memory_fast_path

    def run_task(
''',
    )
    replace_once(
        "src/agentipc/runtime/orchestrator.py",
        '''        if ctx.use_state and not isinstance(ctx.state_hub, StateHub):
            raise _StateHubConfigurationError(
                "use_state=True requires ctx.state_hub to be a StateHub"
            )

        plan_state_ref = None
''',
        '''        if ctx.use_state and not isinstance(ctx.state_hub, StateHub):
            raise _StateHubConfigurationError(
                "use_state=True requires ctx.state_hub to be a StateHub"
            )

        if self._memory_fast_path and ctx.use_memory:
            fast_result = self._try_memory_fast_path(task=task, ctx=ctx)
            if fast_result is not None:
                return fast_result

        plan_state_ref = None
''',
    )
    replace_once(
        "src/agentipc/runtime/orchestrator.py",
        '''    def _dispatch(
        self,
        request: AgentEnvelope,
        ctx: RunContext,
    ) -> AgentEnvelope:
''',
        '''    def _try_memory_fast_path(
        self,
        *,
        task: str,
        ctx: RunContext,
    ) -> AgentEnvelope | None:
        record = ctx.memory_service.get_exact_validated_result(task)
        if record is None:
            return None

        historical_answer = record.payload.get("answer")
        if type(historical_answer) is not str or historical_answer == "":
            return None

        cached_execution = record.payload.get("execution")
        if type(cached_execution) is not dict:
            return None
        operation = cached_execution.get("operation")
        if operation == "identity":
            if "output" not in cached_execution:
                return None
        elif operation == "codeact":
            if not _is_reusable_codeact_execution(cached_execution):
                return None
        else:
            return None

        ctx.memory_service.mark_used(record.memory_id, effective=True)
        if isinstance(ctx.metrics, MetricsCollector):
            ctx.metrics.increment("memory_retrieved")
            ctx.metrics.increment("memory_used")
            ctx.metrics.increment("memory_effective")
            ctx.metrics.increment("fast_path_hit_count")

        evidence_summary = record.payload.get("evidence_summary")
        if type(evidence_summary) is not str:
            evidence_summary = "Reused exact evaluator-validated RESULT memory."

        return AgentEnvelope(
            trace_id=ctx.trace_id,
            task_id=ctx.task_id,
            step_id="step-memory-fast-path",
            sender=_RUNTIME_ID,
            receiver=_RUNTIME_ID,
            message_type=MessageType.RESULT,
            action=ActionType.SUMMARIZE,
            capability="memory_fast_path",
            result={
                "answer": historical_answer,
                "evidence_summary": evidence_summary,
                "fast_path": True,
            },
            status=MessageStatus.OK,
            state_refs=[],
            artifact_refs=[],
            memory_refs=[
                MemoryRef(
                    memory_id=record.memory_id,
                    score=1.0,
                    match_type="exact_validated",
                    summary=record.summary,
                )
            ],
        )

    def _dispatch(
        self,
        request: AgentEnvelope,
        ctx: RunContext,
    ) -> AgentEnvelope:
''',
    )
    replace_once(
        "src/agentipc/runtime/orchestrator.py",
        '''    @staticmethod
    def _record_structured_envelope(
        envelope: AgentEnvelope,
        ctx: RunContext,
    ) -> None:
        if isinstance(ctx.metrics, MetricsCollector):
            payload = encode(envelope)
            ctx.metrics.increment("message_count")
            ctx.metrics.increment("protocol_bytes", len(payload))
            if envelope.state_refs:
                ctx.metrics.increment(
                    "state_transfer_count",
                    len(envelope.state_refs),
                )
                ctx.metrics.increment(
                    "state_bytes",
                    sum(ref.nbytes for ref in envelope.state_refs),
                )
            if envelope.artifact_refs:
                ctx.metrics.increment(
                    "artifact_ref_count",
                    len(envelope.artifact_refs),
                )

        ctx.trace_logger.log_envelope(envelope)
''',
        '''    def _record_structured_envelope(
        self,
        envelope: AgentEnvelope,
        ctx: RunContext,
    ) -> None:
        if isinstance(ctx.metrics, MetricsCollector):
            payload = encode(envelope)
            wire_text = payload.decode("utf-8")
            count = self._wire_counter.count(wire_text)
            ctx.metrics.increment("message_count")
            ctx.metrics.increment("wire_chars", count.text_chars)
            ctx.metrics.increment("wire_tokens", count.text_tokens)
            ctx.metrics.increment("wire_bytes", len(payload))
            ctx.metrics.increment("protocol_bytes", len(payload))
            if envelope.state_refs:
                ctx.metrics.increment(
                    "state_transfer_count",
                    len(envelope.state_refs),
                )
                ctx.metrics.increment(
                    "state_bytes",
                    sum(ref.nbytes for ref in envelope.state_refs),
                )
            if envelope.artifact_refs:
                ctx.metrics.increment(
                    "artifact_ref_count",
                    len(envelope.artifact_refs),
                )
                ctx.metrics.increment(
                    "artifact_payload_bytes",
                    sum(ref.size_bytes for ref in envelope.artifact_refs),
                )

        ctx.trace_logger.log_envelope(envelope)
''',
    )

    replace_once(
        "src/agentipc/evaluation/runner.py",
        '''def run_single(
    *,
    experiment: ExperimentConfig,
    task: str,
    ctx: RunContext,
    agent_registry: AgentRegistry,
) -> RawRunRecord:
''',
        '''def run_single(
    *,
    experiment: ExperimentConfig,
    task: str,
    ctx: RunContext,
    agent_registry: AgentRegistry,
    memory_fast_path: bool = False,
) -> RawRunRecord:
''',
    )
    replace_once(
        "src/agentipc/evaluation/runner.py",
        '''    if not isinstance(agent_registry, AgentRegistry):
        raise TypeError("agent_registry must be an AgentRegistry")

    # Compute task hash (exact UTF-8 bytes, no normalization)
''',
        '''    if not isinstance(agent_registry, AgentRegistry):
        raise TypeError("agent_registry must be an AgentRegistry")
    if type(memory_fast_path) is not bool:
        raise TypeError("memory_fast_path must be a bool")

    # Compute task hash (exact UTF-8 bytes, no normalization)
''',
    )
    replace_once(
        "src/agentipc/evaluation/runner.py",
        '''    router = Router(agent_registry)

    if run_ctx.mode.value == "text":
''',
        '''    router = Router(agent_registry)
    wire_counter = TextCounter()
    # Warm tokenizer outside the measured task timer so instrumentation startup
    # does not bias C/D against a zero-wire fast-path hit.
    wire_counter.count("")

    if run_ctx.mode.value == "text":
''',
    )
    replace_once(
        "src/agentipc/evaluation/runner.py",
        '''        text_transport = TextTransport(
            text_counter=TextCounter(),
            resolver=resolver,
        )
''',
        '''        text_transport = TextTransport(
            text_counter=wire_counter,
            resolver=resolver,
        )
''',
    )
    replace_once(
        "src/agentipc/evaluation/runner.py",
        '''        orchestrator = Orchestrator(router, text_transport=text_transport)
    else:
        # STRUCTURED mode uses direct routing
        orchestrator = Orchestrator(router)
''',
        '''        orchestrator = Orchestrator(
            router,
            text_transport=text_transport,
            wire_counter=wire_counter,
            memory_fast_path=memory_fast_path,
        )
    else:
        # STRUCTURED mode uses direct routing
        orchestrator = Orchestrator(
            router,
            wire_counter=wire_counter,
            memory_fast_path=memory_fast_path,
        )
''',
    )

    write(
        "src/agentipc/experiments/formal/aggregation.py",
        '''from __future__ import annotations

from collections import defaultdict
from typing import Any


FIELDS = (
    "experiment",
    "repeat",
    "latency_ms",
    "message_count",
    "text_chars",
    "text_tokens",
    "wire_chars",
    "wire_tokens",
    "wire_bytes",
    "protocol_bytes",
    "state_transfer_count",
    "state_bytes",
    "artifact_ref_count",
    "artifact_payload_bytes",
    "memory_retrieved",
    "memory_used",
    "memory_effective",
    "memory_harmful",
    "fast_path_hit_count",
    "tool_call_count",
    "repeated_tool_call_count",
    "llm_call_count",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
    "llm_usage_missing_count",
    "llm_latency_ms",
    "success",
)


def flatten_record(record: Any, repeat: int = 0) -> dict[str, Any]:
    exp = getattr(record, "experiment", None)
    name = getattr(exp, "name", exp)
    name = getattr(name, "value", name)
    result = getattr(record, "run_result", record)
    metrics = getattr(result, "metrics", {}) or {}
    if hasattr(metrics, "model_dump"):
        metrics = metrics.model_dump()
    out = {"experiment": str(name), "repeat": repeat}
    for key in FIELDS[2:]:
        out[key] = metrics.get(key, False if key == "success" else 0)
    out["memory_harmful"] = int(out["memory_harmful"])
    out["success"] = bool(out["success"])
    return out


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("experiment") not in {"A", "B", "C", "D"}:
            raise ValueError("invalid experiment")
        grouped[row["experiment"]].append(row)

    summary: dict[str, dict[str, float]] = {}
    for name in ("A", "B", "C", "D"):
        rows = grouped.get(name, [])
        if not rows:
            raise ValueError(f"missing experiment {name}")
        n = len(rows)

        def mean(key: str) -> float:
            return sum(float(row.get(key, 0)) for row in rows) / n

        summary[name] = {
            "mean_latency_ms": mean("latency_ms"),
            "mean_message_count": mean("message_count"),
            "mean_text_chars": mean("text_chars"),
            "mean_text_tokens": mean("text_tokens"),
            "mean_wire_chars": mean("wire_chars"),
            "mean_wire_tokens": mean("wire_tokens"),
            "mean_wire_bytes": mean("wire_bytes"),
            "mean_protocol_bytes": mean("protocol_bytes"),
            "mean_state_transfer_count": mean("state_transfer_count"),
            "mean_state_bytes": mean("state_bytes"),
            "mean_artifact_ref_count": mean("artifact_ref_count"),
            "mean_artifact_payload_bytes": mean("artifact_payload_bytes"),
            "mean_tool_calls": mean("tool_call_count"),
            "mean_repeated_tool_calls": mean("repeated_tool_call_count"),
            "mean_llm_calls": mean("llm_call_count"),
            "mean_llm_prompt_tokens": mean("llm_prompt_tokens"),
            "mean_llm_completion_tokens": mean("llm_completion_tokens"),
            "mean_tokens": mean("llm_total_tokens"),
            "mean_llm_latency_ms": mean("llm_latency_ms"),
            "mean_memory_retrieved": mean("memory_retrieved"),
            "mean_memory_used": mean("memory_used"),
            "memory_effective_rate": sum(bool(row.get("memory_effective")) for row in rows) / n,
            "memory_harmful_rate": sum(bool(row.get("memory_harmful")) for row in rows) / n,
            "success_rate": sum(bool(row.get("success")) for row in rows) / n,
        }
    return summary
''',
    )

    ensure_absent("src/agentipc/experiments/formal/e5_communication.py")
    write(
        "src/agentipc/experiments/formal/e5_communication.py",
        '''from __future__ import annotations

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
            "context=deterministic communication payload;status=ok\\n"
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
''',
    )

    ensure_absent("src/agentipc/experiments/formal/e6_memory_fast_path.py")
    write(
        "src/agentipc/experiments/formal/e6_memory_fast_path.py",
        '''from __future__ import annotations

from pathlib import Path
from typing import Any

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.experiment import EXPERIMENT_C, EXPERIMENT_D, ExperimentConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.runner import run_single
from agentipc.evaluation.text_counter import TextCounter
from agentipc.evaluation.trace import TraceLogger
from agentipc.experiments.real_bailian.config import load_real_bailian_config
from agentipc.experiments.real_bailian.evaluate import evaluate_normalized_knowledge_answer
from agentipc.experiments.real_bailian.recording_executor import RecordingExecutorAgent
from agentipc.experiments.real_bailian.runner import build_recording_provider_bundle
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext
from agentipc.scenarios.codeact_chain import _build_runtime_task, _validate_fixture_root
from agentipc.scenarios.codeact_eval import evaluate_codeact_execution
from agentipc.scenarios.knowledge_loader import load_knowledge_documents
from agentipc.scenarios.models import load_codeact_tasks, load_knowledge_tasks
from agentipc.state.hub import StateHub


_CONFIGS: tuple[tuple[str, ExperimentConfig, bool], ...] = (
    ("C", EXPERIMENT_C, False),
    ("D", EXPERIMENT_D, False),
    ("D-Fast", EXPERIMENT_D, True),
)


def run_e6(
    *,
    root: Path,
    result_dir: Path,
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(repeat) is not int or repeat < 1:
        raise ValueError("repeat must be >= 1")
    _require_wire_tokenizer()

    secret = load_real_bailian_config()
    llm_recorder, embedding_recorder = build_recording_provider_bundle(secret)
    provider_bundle = ProviderBundle(llm=llm_recorder, embedding=embedding_recorder)

    knowledge_tasks = load_knowledge_tasks(root / "scenarios/knowledge_chain/tasks.json")
    knowledge_base = [task for task in knowledge_tasks if 1 <= task.round <= 5]
    if [task.round for task in knowledge_base] != [1, 2, 3, 4, 5]:
        raise ValueError("E6 Knowledge workload requires rounds 1..5")
    documents = load_knowledge_documents(root / "scenarios/knowledge_chain/knowledge")
    knowledge_data = [
        {
            "document_id": document.document_id,
            "text": document.body,
            "keywords": list(document.tags),
        }
        for document in documents
    ]

    codeact_tasks = load_codeact_tasks(root / "scenarios/codeact_chain/tasks.json")
    codeact_base = [task for task in codeact_tasks if 1 <= task.round <= 5]
    if [task.round for task in codeact_base] != [1, 2, 3, 4, 5]:
        raise ValueError("E6 CodeAct workload requires rounds 1..5")
    fixture_root = _validate_fixture_root(root / "scenarios/codeact_chain")
    codeact_runtime = {
        task.round: _build_runtime_task(task, fixture_root)
        for task in codeact_base
    }

    rows: list[dict[str, Any]] = []
    for repeat_index in range(repeat):
        repeat_number = repeat_index + 1
        runtime_config = AgentIPCConfig(
            llm_provider="openai",
            embedding_provider="openai",
            random_seed=42 + repeat_index,
        )
        repeat_root = result_dir / f"repeat-{repeat_number:03d}" / "e6"
        for config_name, experiment, fast_path in _CONFIGS:
            rows.extend(
                _run_knowledge_config(
                    config_name=config_name,
                    experiment=experiment,
                    fast_path=fast_path,
                    tasks=knowledge_base,
                    knowledge=knowledge_data,
                    runtime_config=runtime_config,
                    provider_bundle=provider_bundle,
                    repeat_number=repeat_number,
                    work_root=repeat_root / "knowledge" / config_name.lower().replace("-", "_"),
                )
            )
            rows.extend(
                _run_codeact_config(
                    config_name=config_name,
                    experiment=experiment,
                    fast_path=fast_path,
                    tasks=codeact_base,
                    runtime_tasks=codeact_runtime,
                    runtime_config=runtime_config,
                    provider_bundle=provider_bundle,
                    repeat_number=repeat_number,
                    work_root=repeat_root / "codeact" / config_name.lower().replace("-", "_"),
                )
            )

    return rows, _aggregate_e6(rows, repeat=repeat)


def _run_knowledge_config(
    *,
    config_name: str,
    experiment: ExperimentConfig,
    fast_path: bool,
    tasks,
    knowledge: list[dict[str, object]],
    runtime_config: AgentIPCConfig,
    provider_bundle: ProviderBundle,
    repeat_number: int,
    work_root: Path,
) -> list[dict[str, Any]]:
    work_root.mkdir(parents=True, exist_ok=False)
    (work_root / "traces").mkdir()
    store = SQLiteMemoryStore(work_root / "memory")
    rows: list[dict[str, Any]] = []
    try:
        memory_service = MemoryService(
            store,
            provider_bundle.embedding,
            VectorIndex(provider_bundle.embedding.dim),
        )
        schedule = [(task, False) for task in tasks] + [(task, True) for task in tasks]
        for sequence, (task, is_repeat) in enumerate(schedule, start=1):
            state_hub = StateHub(transport="shm")
            try:
                metrics = MetricsCollector()
                task_id = f"e6-knowledge-r{repeat_number}-{config_name}-{sequence:02d}"
                ctx = RunContext(
                    trace_id=f"{task_id}-trace",
                    task_id=task_id,
                    mode=experiment.mode,
                    config=runtime_config,
                    registry=CapabilityRegistry(),
                    state_hub=state_hub,
                    artifact_store=ArtifactStore(work_root / "artifacts" / f"task-{sequence:02d}"),
                    memory_service=memory_service,
                    metrics=metrics,
                    trace_logger=TraceLogger(work_root / "traces" / f"task-{sequence:02d}.jsonl"),
                    provider_bundle=provider_bundle,
                    use_state=experiment.use_state,
                    use_memory=experiment.use_memory,
                    use_sandbox=False,
                )
                agents = AgentRegistry()
                agents.register(PlannerAgent())
                agents.register(RetrieverAgent(knowledge=knowledge))
                agents.register(ExecutorAgent())
                agents.register(SummarizerAgent())

                record = run_single(
                    experiment=experiment,
                    task=task.query,
                    ctx=ctx,
                    agent_registry=agents,
                    memory_fast_path=fast_path,
                )
                evaluation = evaluate_normalized_knowledge_answer(task, record.run_result.answer)
                _persist_validation_if_written(
                    memory_service,
                    task_id=task_id,
                    passed=evaluation.success,
                    enabled=experiment.use_memory,
                )
                rows.append(
                    _row(
                        group="knowledge",
                        config_name=config_name,
                        repeat_number=repeat_number,
                        sequence=sequence,
                        source_round=task.round,
                        is_repeat=is_repeat,
                        task_hash=record.task_hash,
                        runtime_success=record.run_result.success,
                        evaluation_pass=evaluation.success,
                        metrics=dict(record.run_result.metrics),
                    )
                )
            finally:
                state_hub.close()
    finally:
        store.close()
    return rows


def _run_codeact_config(
    *,
    config_name: str,
    experiment: ExperimentConfig,
    fast_path: bool,
    tasks,
    runtime_tasks: dict[int, str],
    runtime_config: AgentIPCConfig,
    provider_bundle: ProviderBundle,
    repeat_number: int,
    work_root: Path,
) -> list[dict[str, Any]]:
    work_root.mkdir(parents=True, exist_ok=False)
    (work_root / "traces").mkdir()
    store = SQLiteMemoryStore(work_root / "memory")
    rows: list[dict[str, Any]] = []
    try:
        memory_service = MemoryService(
            store,
            provider_bundle.embedding,
            VectorIndex(provider_bundle.embedding.dim),
        )
        schedule = [(task, False) for task in tasks] + [(task, True) for task in tasks]
        for sequence, (task, is_repeat) in enumerate(schedule, start=1):
            runtime_task = runtime_tasks[task.round]
            state_hub = StateHub(transport="shm")
            recorder = RecordingExecutorAgent()
            try:
                metrics = MetricsCollector()
                task_id = f"e6-codeact-r{repeat_number}-{config_name}-{sequence:02d}"
                ctx = RunContext(
                    trace_id=f"{task_id}-trace",
                    task_id=task_id,
                    mode=experiment.mode,
                    config=runtime_config,
                    registry=CapabilityRegistry(),
                    state_hub=state_hub,
                    artifact_store=ArtifactStore(work_root / "artifacts" / f"task-{sequence:02d}"),
                    memory_service=memory_service,
                    metrics=metrics,
                    trace_logger=TraceLogger(work_root / "traces" / f"task-{sequence:02d}.jsonl"),
                    provider_bundle=provider_bundle,
                    use_state=experiment.use_state,
                    use_memory=experiment.use_memory,
                    use_sandbox=True,
                )
                agents = AgentRegistry()
                agents.register(PlannerAgent())
                agents.register(RetrieverAgent(knowledge=[]))
                agents.register(recorder)
                agents.register(SummarizerAgent())

                record = run_single(
                    experiment=experiment,
                    task=runtime_task,
                    ctx=ctx,
                    agent_registry=agents,
                    memory_fast_path=fast_path,
                )
                execution = recorder.executions[-1] if recorder.executions else None
                if execution is None and record.run_result.metrics.get("fast_path_hit_count", 0):
                    cached = memory_service.get_exact_validated_result(runtime_task)
                    if cached is not None:
                        maybe_execution = cached.payload.get("execution")
                        if type(maybe_execution) is dict:
                            execution = maybe_execution
                evaluation_pass = bool(
                    type(execution) is dict
                    and evaluate_codeact_execution(task, execution).success
                )
                _persist_validation_if_written(
                    memory_service,
                    task_id=task_id,
                    passed=evaluation_pass,
                    enabled=experiment.use_memory,
                )
                rows.append(
                    _row(
                        group="codeact",
                        config_name=config_name,
                        repeat_number=repeat_number,
                        sequence=sequence,
                        source_round=task.round,
                        is_repeat=is_repeat,
                        task_hash=record.task_hash,
                        runtime_success=record.run_result.success,
                        evaluation_pass=evaluation_pass,
                        metrics=dict(record.run_result.metrics),
                    )
                )
            finally:
                state_hub.close()
    finally:
        store.close()
    return rows


def _persist_validation_if_written(
    memory_service: MemoryService,
    *,
    task_id: str,
    passed: bool,
    enabled: bool,
) -> None:
    if not enabled:
        return
    memory_id = f"mem_{task_id}"
    if memory_service.get(memory_id) is not None:
        memory_service.mark_validated_result(memory_id, passed=passed)


def _row(
    *,
    group: str,
    config_name: str,
    repeat_number: int,
    sequence: int,
    source_round: int,
    is_repeat: bool,
    task_hash: str,
    runtime_success: bool,
    evaluation_pass: bool,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    fast_hit = int(metrics.get("fast_path_hit_count", 0))
    return {
        "experiment": "E6",
        "group": group,
        "config": config_name,
        "repeat": repeat_number,
        "sequence": sequence,
        "phase": "repeat" if is_repeat else "new",
        "source_round": source_round,
        "task_hash": task_hash,
        "runtime_success": runtime_success,
        "evaluation_pass": evaluation_pass,
        "validated_fast_path_effective": int(bool(fast_hit and evaluation_pass)),
        "validated_fast_path_harmful": int(bool(fast_hit and not evaluation_pass)),
        "metrics": metrics,
    }


def _aggregate_e6(rows: list[dict[str, Any]], *, repeat: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in ("knowledge", "codeact"):
        group_rows = [row for row in rows if row["group"] == group]
        configs = {
            name: _aggregate_rows([row for row in group_rows if row["config"] == name])
            for name, _, _ in _CONFIGS
        }
        d_fast_rows = [row for row in group_rows if row["config"] == "D-Fast"]
        groups[group] = {
            "configs": configs,
            "c_vs_d_fast": _delta(configs["C"], configs["D-Fast"]),
            "d_vs_d_fast": _delta(configs["D"], configs["D-Fast"]),
            "zero_repeat_control": _aggregate_rows(
                [row for row in d_fast_rows if row["phase"] == "new"]
            ),
            "repeat_half": _aggregate_rows(
                [row for row in d_fast_rows if row["phase"] == "repeat"]
            ),
        }

    overall_configs = {
        name: _aggregate_rows([row for row in rows if row["config"] == name])
        for name, _, _ in _CONFIGS
    }
    overall_fast = [row for row in rows if row["config"] == "D-Fast"]
    zero_control = _aggregate_rows([row for row in overall_fast if row["phase"] == "new"])
    repeat_half = _aggregate_rows([row for row in overall_fast if row["phase"] == "repeat"])
    passed = bool(
        all(item["evaluation_pass_rate"] == 1.0 for item in overall_configs.values())
        and zero_control["total_fast_path_hits"] == 0
        and repeat_half["validated_fast_path_harmful_count"] == 0
    )
    return {
        "experiment": "E6",
        "repeat": repeat,
        "workload": "per group: 5 new tasks + the exact same 5 tasks repeated",
        "repeat_ratio": 0.5,
        "match_method": "exact task_topic + evaluator-validated RESULT",
        "passed": passed,
        "groups": groups,
        "overall": {
            "configs": overall_configs,
            "c_vs_d_fast": _delta(overall_configs["C"], overall_configs["D-Fast"]),
            "d_vs_d_fast": _delta(overall_configs["D"], overall_configs["D-Fast"]),
            "zero_repeat_control": zero_control,
            "repeat_half": repeat_half,
        },
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty E6 row set")
    n = len(rows)

    def total(metric: str) -> float:
        return sum(float(row["metrics"].get(metric, 0)) for row in rows)

    fast_hits = int(total("fast_path_hit_count"))
    validated_effective = sum(int(row["validated_fast_path_effective"]) for row in rows)
    validated_harmful = sum(int(row["validated_fast_path_harmful"]) for row in rows)
    return {
        "task_count": n,
        "runtime_success_rate": sum(bool(row["runtime_success"]) for row in rows) / n,
        "evaluation_pass_rate": sum(bool(row["evaluation_pass"]) for row in rows) / n,
        "total_llm_prompt_tokens": int(total("llm_prompt_tokens")),
        "total_llm_completion_tokens": int(total("llm_completion_tokens")),
        "total_llm_tokens": int(total("llm_total_tokens")),
        "total_llm_calls": int(total("llm_call_count")),
        "total_wire_tokens": int(total("wire_tokens")),
        "total_wire_bytes": int(total("wire_bytes")),
        "total_tool_calls": int(total("tool_call_count")),
        "total_memory_retrieved": int(total("memory_retrieved")),
        "total_memory_used": int(total("memory_used")),
        "total_memory_effective": int(total("memory_effective")),
        "total_memory_harmful": int(total("memory_harmful")),
        "total_fast_path_hits": fast_hits,
        "fast_path_hit_rate": fast_hits / n,
        "validated_fast_path_effective_count": validated_effective,
        "validated_fast_path_harmful_count": validated_harmful,
        "validated_fast_path_effective_rate": validated_effective / fast_hits if fast_hits else 0.0,
        "wrong_harmful_memory_rate": validated_harmful / fast_hits if fast_hits else 0.0,
        "total_latency_ms": total("latency_ms"),
        "mean_latency_ms": total("latency_ms") / n,
    }


def _delta(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, float]:
    return {
        "provider_prompt_token_saving_pct": _reduction_pct(
            baseline["total_llm_prompt_tokens"], current["total_llm_prompt_tokens"]
        ),
        "provider_total_token_saving_pct": _reduction_pct(
            baseline["total_llm_tokens"], current["total_llm_tokens"]
        ),
        "llm_call_reduction_pct": _reduction_pct(
            baseline["total_llm_calls"], current["total_llm_calls"]
        ),
        "wire_token_saving_pct": _reduction_pct(
            baseline["total_wire_tokens"], current["total_wire_tokens"]
        ),
        "wire_byte_saving_pct": _reduction_pct(
            baseline["total_wire_bytes"], current["total_wire_bytes"]
        ),
        "tool_call_reduction_pct": _reduction_pct(
            baseline["total_tool_calls"], current["total_tool_calls"]
        ),
        "latency_reduction_pct": _reduction_pct(
            baseline["total_latency_ms"], current["total_latency_ms"]
        ),
    }


def _reduction_pct(baseline: float, current: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return (baseline - current) / baseline * 100.0


def _require_wire_tokenizer() -> None:
    probe = TextCounter().count("AgentIPC E6 tokenizer probe.")
    if probe.token_method == "unavailable":
        raise RuntimeError(
            "E6 requires tiktoken for wire-token accounting; "
            "install with: pip install -e '.[tiktoken]'"
        )
''',
    )

    replace_once(
        "src/agentipc/experiments/formal/report.py",
        '''    if summary.get("experiment") in {"E2", "E3"}:
        return _render_continuous_task(summary)
    if "inproc" in summary and "shm" in summary:
''',
        '''    if summary.get("experiment") in {"E2", "E3"}:
        return _render_continuous_task(summary)
    if summary.get("experiment") == "E5":
        return _render_e5(summary)
    if summary.get("experiment") == "E6":
        return _render_e6(summary)
    if "inproc" in summary and "shm" in summary:
''',
    )
    replace_once(
        "src/agentipc/experiments/formal/report.py",
        '''def _render_e4(summary: dict[str, Any]) -> str:
''',
        '''def _render_e5(summary: dict[str, Any]) -> str:
    lines = [
        "# E5 Communication Cost Benchmark",
        "",
        f"Tokenizer: `{summary['token_method']}`",
        "",
        "| Payload | Mode | Wire chars | Wire tokens | Wire bytes | Artifact payload bytes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, payload in sorted(summary["payloads"].items(), key=lambda item: int(item[0])):
        size = payload["payload_bytes"]
        for mode in ("A", "B", "C"):
            item = payload["modes"][mode]
            lines.append(
                f"| {size} | {mode} | {item['mean_wire_chars']:.2f} | "
                f"{item['mean_wire_tokens']:.2f} | {item['mean_wire_bytes']:.2f} | "
                f"{item['mean_artifact_payload_bytes']:.2f} |"
            )
        lines.append("")
        lines.append(
            f"- {size} B vs A: token {payload['savings_vs_A']['B']['wire_token_saving_pct']:.2f}%, "
            f"byte {payload['savings_vs_A']['B']['wire_byte_saving_pct']:.2f}%"
        )
        lines.append(
            f"- {size} C vs A: token {payload['savings_vs_A']['C']['wire_token_saving_pct']:.2f}%, "
            f"byte {payload['savings_vs_A']['C']['wire_byte_saving_pct']:.2f}%, "
            f"total-transfer {payload['savings_vs_A']['C']['total_transfer_saving_pct']:.2f}%"
        )
        lines.append("")
    lines.append(summary["note"])
    return "\\n".join(lines) + "\\n"


def _render_e6(summary: dict[str, Any]) -> str:
    lines = [
        "# E6 Real Provider Memory Fast Path",
        "",
        f"Overall pass: **{summary['passed']}**",
        f"Workload: {summary['workload']}",
        f"Match: {summary['match_method']}",
        "",
    ]
    for group in ("knowledge", "codeact"):
        data = summary["groups"][group]
        lines.extend([
            f"## {group.title()}",
            "",
            "| Config | Eval pass | Provider tokens | LLM calls | Wire tokens | Tool calls | Mean latency ms | Fast hits | Harmful rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in ("C", "D", "D-Fast"):
            item = data["configs"][name]
            lines.append(
                f"| {name} | {item['evaluation_pass_rate']:.1%} | "
                f"{item['total_llm_tokens']} | {item['total_llm_calls']} | "
                f"{item['total_wire_tokens']} | {item['total_tool_calls']} | "
                f"{item['mean_latency_ms']:.3f} | {item['total_fast_path_hits']} | "
                f"{item['wrong_harmful_memory_rate']:.1%} |"
            )
        delta = data["c_vs_d_fast"]
        lines.extend([
            "",
            "C vs D-Fast:",
            f"- Provider prompt token saving: {delta['provider_prompt_token_saving_pct']:.2f}%",
            f"- Provider total token saving: {delta['provider_total_token_saving_pct']:.2f}%",
            f"- LLM call reduction: {delta['llm_call_reduction_pct']:.2f}%",
            f"- Wire token saving: {delta['wire_token_saving_pct']:.2f}%",
            f"- Latency reduction: {delta['latency_reduction_pct']:.2f}%",
            f"- 0% repeat control fast hits: {data['zero_repeat_control']['total_fast_path_hits']}",
            "",
        ])
    return "\\n".join(lines) + "\\n"


def _render_e4(summary: dict[str, Any]) -> str:
''',
    )

    replace_once(
        "scripts/run_formal_experiment.py",
        '''    parser.add_argument("experiment", choices=["e1", "e2", "e3", "e4"])
''',
        '''    parser.add_argument(
        "experiment",
        choices=["e1", "e2", "e3", "e4", "e5", "e6"],
    )
''',
    )
    replace_once(
        "scripts/run_formal_experiment.py",
        '''        args.experiment in {"e2", "e3"}
        or (args.experiment == "e1" and args.provider == "openai")
    )
    if args.experiment in {"e2", "e3"} and args.provider != "openai":
        parser.error("e2/e3 require --provider openai")
''',
        '''        args.experiment in {"e2", "e3", "e6"}
        or (args.experiment == "e1" and args.provider == "openai")
    )
    if args.experiment in {"e2", "e3", "e6"} and args.provider != "openai":
        parser.error("e2/e3/e6 require --provider openai")
''',
    )
    replace_once(
        "scripts/run_formal_experiment.py",
        '''        "e4": "e4-shm",
    }
''',
        '''        "e4": "e4-shm",
        "e5": "e5-communication",
        "e6": "e6-memory-fast-path",
    }
''',
    )
    replace_once(
        "scripts/run_formal_experiment.py",
        '''    elif args.experiment == "e1":
        from agentipc.experiments.formal.factory import build_factory
''',
        '''    elif args.experiment == "e5":
        from agentipc.experiments.formal.e5_communication import run_e5

        rows, summary = run_e5(root=root / "work", repeat=args.repeat)
    elif args.experiment == "e6":
        from agentipc.experiments.formal.e6_memory_fast_path import run_e6

        rows, summary = run_e6(
            root=Path(".").resolve(),
            result_dir=root,
            repeat=args.repeat,
        )
    elif args.experiment == "e1":
        from agentipc.experiments.formal.factory import build_factory
''',
    )

    for rel in (
        "tests/evaluation/test_metrics_model.py",
        "tests/evaluation/test_metrics_collector.py",
    ):
        replace_once(
            rel,
            '''    "text_chars",
    "text_tokens",
    "protocol_bytes",
''',
            '''    "text_chars",
    "text_tokens",
    "wire_chars",
    "wire_tokens",
    "wire_bytes",
    "protocol_bytes",
''',
        )
        replace_once(
            rel,
            '''    "artifact_ref_count",
    "memory_retrieved",
''',
            '''    "artifact_ref_count",
    "artifact_payload_bytes",
    "memory_retrieved",
''',
        )
        replace_once(
            rel,
            '''    "memory_effective",
    "memory_harmful",
    "tool_call_count",
''',
            '''    "memory_effective",
    "memory_harmful",
    "fast_path_hit_count",
    "tool_call_count",
''',
        )

    ensure_absent("tests/runtime/test_memory_fast_path.py")
    write(
        "tests/runtime/test_memory_fast_path.py",
        '''from __future__ import annotations

from pathlib import Path

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router
from agentipc.state.hub import StateHub


TASK = "exact repeated task"
ANSWER = "validated cached answer"


def _agents() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(PlannerAgent())
    registry.register(RetrieverAgent(knowledge=[{
        "document_id": "doc-1",
        "text": "exact repeated task evidence",
        "keywords": ["exact", "repeated"],
    }]))
    registry.register(ExecutorAgent())
    registry.register(SummarizerAgent())
    return registry


def _bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={"exact repeated task": ANSWER},
            default_text=ANSWER,
        ),
        embedding=HashEmbeddingProvider(dim=32),
    )


def _ctx(tmp_path: Path, *, name: str, service: MemoryService, bundle: ProviderBundle):
    metrics = MetricsCollector()
    hub = StateHub(transport="inproc")
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(),
        registry=CapabilityRegistry(),
        state_hub=hub,
        artifact_store=ArtifactStore(tmp_path / f"artifacts-{name}"),
        memory_service=service,
        metrics=metrics,
        trace_logger=TraceLogger(tmp_path / f"{name}.jsonl"),
        provider_bundle=bundle,
        use_state=False,
        use_memory=True,
        use_sandbox=False,
    )
    return metrics, hub, ctx


def test_exact_validated_result_skips_full_pipeline(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    embedding = HashEmbeddingProvider(dim=32)
    service = MemoryService(store, embedding, VectorIndex(embedding.dim))
    bundle = _bundle()
    try:
        first_metrics, first_hub, first_ctx = _ctx(
            tmp_path, name="first", service=service, bundle=bundle
        )
        try:
            first = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=first_ctx)
        finally:
            first_hub.close()
        assert first.result is not None
        assert first.result["answer"] == ANSWER
        assert first_metrics.snapshot().llm_call_count == 2
        service.mark_validated_result("mem_task-first", passed=True)

        second_metrics, second_hub, second_ctx = _ctx(
            tmp_path, name="second", service=service, bundle=bundle
        )
        try:
            second = Orchestrator(
                Router(_agents()), memory_fast_path=True
            ).run_task(task=TASK, ctx=second_ctx)
        finally:
            second_hub.close()
        assert second.result is not None
        assert second.result["answer"] == ANSWER
        snapshot = second_metrics.snapshot()
        assert snapshot.fast_path_hit_count == 1
        assert snapshot.llm_call_count == 0
        assert snapshot.message_count == 0
        assert snapshot.tool_call_count == 0
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 1
        assert snapshot.memory_effective == 1
        assert snapshot.memory_harmful == 0
        assert service.get("mem_task-second") is None
    finally:
        store.close()


def test_unvalidated_result_does_not_take_fast_path(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    embedding = HashEmbeddingProvider(dim=32)
    service = MemoryService(store, embedding, VectorIndex(embedding.dim))
    bundle = _bundle()
    try:
        first_metrics, first_hub, first_ctx = _ctx(
            tmp_path, name="unvalidated-first", service=service, bundle=bundle
        )
        try:
            Orchestrator(Router(_agents())).run_task(task=TASK, ctx=first_ctx)
        finally:
            first_hub.close()
        assert first_metrics.snapshot().llm_call_count == 2

        second_metrics, second_hub, second_ctx = _ctx(
            tmp_path, name="unvalidated-second", service=service, bundle=bundle
        )
        try:
            Orchestrator(
                Router(_agents()), memory_fast_path=True
            ).run_task(task=TASK, ctx=second_ctx)
        finally:
            second_hub.close()
        assert second_metrics.snapshot().fast_path_hit_count == 0
        assert second_metrics.snapshot().llm_call_count == 2
    finally:
        store.close()
''',
    )

    ensure_absent("tests/experiments/test_e5_communication.py")
    write(
        "tests/experiments/test_e5_communication.py",
        '''from __future__ import annotations

from agentipc.evaluation.text_counter import TextCount, TextCounter
from agentipc.experiments.formal.e5_communication import run_e5


class FakeTokenCounter(TextCounter):
    def __init__(self) -> None:
        pass

    def count(self, text: str) -> TextCount:
        return TextCount(
            text_chars=len(text),
            text_tokens=max(1, len(text.encode("utf-8")) // 4),
            token_method="fake:e5-test",
        )


def test_e5_measures_all_modes_and_payload_sizes(tmp_path) -> None:
    rows, summary = run_e5(
        root=tmp_path / "e5",
        repeat=2,
        payload_sizes=(256, 2048),
        text_counter=FakeTokenCounter(),
    )
    assert len(rows) == 12
    assert summary["experiment"] == "E5"
    large = summary["payloads"]["2048"]
    assert set(large["modes"]) == {"A", "B", "C"}
    assert large["modes"]["C"]["mean_artifact_payload_bytes"] == 2048.0
    assert large["modes"]["C"]["mean_wire_bytes"] < large["modes"]["A"]["mean_wire_bytes"]
    assert large["savings_vs_A"]["C"]["wire_byte_saving_pct"] > 0.0
''',
    )

    print("AgentIPC E5/E6 patch applied.")
    print("Next: git diff --check, targeted pytest, then full pytest.")


if __name__ == "__main__":
    main()
