from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from agentipc.evaluation.metrics import MetricsCollector, MetricsSnapshot, TaskTimer
from agentipc.evaluation.text_counter import TextCount, TextCounter
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef


def _observed_fake_send(
    *,
    envelope: AgentEnvelope,
    rendered_text: str,
    collector: MetricsCollector,
    text_counter: TextCounter,
    trace_logger: TraceLogger,
    timer: TaskTimer,
) -> tuple[bytes, TextCount, TraceEvent, MetricsSnapshot]:
    timer.start()

    encoded = encode(envelope)
    text_count = text_counter.count(rendered_text)

    collector.increment("message_count")
    collector.increment("text_chars", text_count.text_chars)
    collector.increment("text_tokens", text_count.text_tokens)
    collector.increment("protocol_bytes", len(encoded))
    collector.increment("state_transfer_count", len(envelope.state_refs))
    collector.increment(
        "state_bytes",
        sum(ref.nbytes for ref in envelope.state_refs),
    )
    collector.increment("artifact_ref_count", len(envelope.artifact_refs))
    collector.increment("memory_retrieved", len(envelope.memory_refs))

    event = trace_logger.log_envelope(envelope)

    elapsed = timer.stop()
    collector.set_latency_ms(elapsed)
    collector.set_success(True)

    return encoded, text_count, event, collector.snapshot()


def test_fake_message_send_produces_metrics_and_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter([1_000_000_000, 1_005_000_000])
    monkeypatch.setattr(
        "agentipc.evaluation.metrics.time.perf_counter_ns",
        lambda: next(clock_values),
    )

    class FakeEncoding:
        def encode(self, text: str) -> list[int]:
            assert text == rendered_text
            return [1, 2, 3, 4]

    monkeypatch.setattr(
        "agentipc.evaluation.text_counter._load_tiktoken",
        lambda: SimpleNamespace(get_encoding=lambda name: FakeEncoding()),
    )

    state_ref = StateRef(
        uri="shm://observability-plan",
        kind="plan_vector",
        shape=[64],
        dtype="float32",
        nbytes=256,
        checksum="observability-checksum",
        transport="shared_memory",
        summary="plan vector",
    )
    artifact_ref = ArtifactRef(
        uri="artifact://sha256/" + "a" * 64,
        sha256="a" * 64,
        media_type="application/json",
        size_bytes=128,
        summary="retrieved evidence",
    )
    memory_ref = MemoryRef(
        memory_id="memory-observed-1",
        score=0.8,
        match_type="hybrid",
        summary="previous troubleshooting result",
    )
    envelope = AgentEnvelope(
        message_id="msg-observed-1",
        trace_id="trace-observed",
        task_id="task-observed",
        step_id="step-retrieve",
        sender="planner",
        receiver="retriever",
        message_type=MessageType.REQUEST,
        action=ActionType.RETRIEVE,
        args={"query": "openEuler network troubleshooting"},
        result={"status": "fake-result"},
        state_refs=[state_ref],
        artifact_refs=[artifact_ref],
        memory_refs=[memory_ref],
        status=MessageStatus.OK,
        metrics={"sentinel": "unchanged"},
    )
    rendered_text = (
        "planner -> retriever: retrieve openEuler "
        "network troubleshooting evidence"
    )
    trace_path = tmp_path / "trace.jsonl"
    collector = MetricsCollector()
    text_counter = TextCounter(encoding_name="test_encoding")
    trace_logger = TraceLogger(trace_path)
    timer = TaskTimer()

    encoded, text_count, logged_event, snapshot = _observed_fake_send(
        envelope=envelope,
        rendered_text=rendered_text,
        collector=collector,
        text_counter=text_counter,
        trace_logger=trace_logger,
        timer=timer,
    )

    assert decode(encoded) == envelope
    assert snapshot.message_count == 1
    assert snapshot.text_chars == len(rendered_text)
    assert snapshot.text_tokens == 4
    assert snapshot.protocol_bytes == len(encoded)
    assert snapshot.state_transfer_count == 1
    assert snapshot.state_bytes == 256
    assert snapshot.artifact_ref_count == 1
    assert snapshot.memory_retrieved == 1
    assert snapshot.memory_used == 0
    assert snapshot.memory_effective == 0
    assert snapshot.memory_harmful == 0
    assert snapshot.tool_call_count == 0
    assert snapshot.repeated_tool_call_count == 0
    assert snapshot.latency_ms == 5.0
    assert snapshot.success is True

    assert text_count.text_chars == len(rendered_text)
    assert text_count.text_tokens == 4
    assert text_count.token_method == "tiktoken:test_encoding"
    assert envelope.metrics == {"sentinel": "unchanged"}

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    trace_event = TraceEvent.model_validate_json(lines[0])
    assert trace_event == logged_event
    assert trace_event.event_type == "message"
    assert trace_event.trace_id == envelope.trace_id
    assert trace_event.task_id == envelope.task_id
    assert trace_event.message_id == envelope.message_id
    assert trace_event.step_id == envelope.step_id
    assert trace_event.state_refs == envelope.state_refs
    assert trace_event.artifact_refs == envelope.artifact_refs
    assert trace_event.memory_refs == envelope.memory_refs
    assert math.isfinite(trace_event.recorded_at)
    assert trace_event.recorded_at >= 0

    trace_payload = json.loads(lines[0])
    assert "args" not in trace_payload
    assert "result" not in trace_payload
    assert "metrics" not in trace_payload

    assert snapshot.message_count == len(lines) == 1
    assert snapshot.state_transfer_count == len(trace_event.state_refs)
    assert snapshot.state_bytes == sum(ref.nbytes for ref in trace_event.state_refs)
    assert snapshot.artifact_ref_count == len(trace_event.artifact_refs)
    assert snapshot.memory_retrieved == len(trace_event.memory_refs)
