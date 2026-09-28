from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef


def _state_ref() -> StateRef:
    return StateRef(
        uri="shm://plan-vector-1",
        kind="plan_vector",
        shape=[3],
        dtype="float32",
        nbytes=12,
        checksum="state-checksum-sentinel",
        transport="shared_memory",
        summary="openEuler 网络恢复",
    )


def _artifact_ref() -> ArtifactRef:
    return ArtifactRef(
        uri="artifact://evidence-1",
        sha256="a" * 64,
        media_type="application/json",
        size_bytes=128,
        summary="证据摘要 sentinel",
    )


def _memory_ref() -> MemoryRef:
    return MemoryRef(
        memory_id="memory-复用-1",
        score=0.95,
        match_type="hybrid",
        summary="历史经验 sentinel",
    )


def _envelope(message_id: str, *, step_id: str) -> AgentEnvelope:
    return AgentEnvelope(
        message_id=message_id,
        trace_id="trace-fixed",
        task_id="task-fixed",
        step_id=step_id,
        sender="规划器-α",
        receiver="检索器-β",
        message_type=MessageType.REQUEST,
        action=ActionType.RETRIEVE,
        args={"secret": "DO_NOT_TRACE_ARGS"},
        result={"secret": "DO_NOT_TRACE_RESULT"},
        state_refs=[_state_ref()],
        artifact_refs=[_artifact_ref()],
        memory_refs=[_memory_ref()],
        status=MessageStatus.OK,
        metrics={"secret": "DO_NOT_TRACE_METRICS"},
    )


def test_multi_event_jsonl_round_trip_and_order(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trace.jsonl"
    logger = TraceLogger(path)
    envelope1 = _envelope("msg-1", step_id="step-1")
    envelope2 = _envelope("msg-2", step_id="step-2")

    event1 = logger.log_envelope(envelope1)
    event2 = logger.log_envelope(envelope2)

    assert logger.path == path
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    parsed = [TraceEvent.model_validate_json(line) for line in lines]
    assert parsed == [event1, event2]
    assert [event.message_id for event in parsed] == ["msg-1", "msg-2"]


def test_identity_enums_and_refs_are_preserved(tmp_path: Path) -> None:
    envelope = _envelope("msg-identity", step_id="step-identity")
    event = TraceLogger(tmp_path / "trace.jsonl").log_envelope(envelope)

    assert event.event_type == "message"
    assert event.trace_id == envelope.trace_id
    assert event.task_id == envelope.task_id
    assert event.message_id == envelope.message_id
    assert event.step_id == envelope.step_id
    assert event.sender == envelope.sender
    assert event.receiver == envelope.receiver
    assert event.message_type == MessageType.REQUEST.value
    assert event.action == ActionType.RETRIEVE.value
    assert event.status == MessageStatus.OK.value
    assert event.state_refs == envelope.state_refs
    assert event.artifact_refs == envelope.artifact_refs
    assert event.memory_refs == envelope.memory_refs
    assert math.isfinite(event.recorded_at)
    assert event.recorded_at >= 0


def test_none_action_serializes_as_json_null(tmp_path: Path) -> None:
    envelope = _envelope("msg-no-action", step_id="step-no-action")
    envelope.action = None
    path = tmp_path / "trace.jsonl"

    TraceLogger(path).log_envelope(envelope)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["action"] is None


def test_trace_excludes_args_result_metrics_and_resolved_payload(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    TraceLogger(path).log_envelope(_envelope("msg-small", step_id="step-small"))

    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert "args" not in payload
    assert "result" not in payload
    assert "metrics" not in payload
    assert "DO_NOT_TRACE_ARGS" not in raw
    assert "DO_NOT_TRACE_RESULT" not in raw
    assert "DO_NOT_TRACE_METRICS" not in raw
    assert payload["state_refs"][0]["uri"] == "shm://plan-vector-1"
    assert payload["artifact_refs"][0]["uri"] == "artifact://evidence-1"
    assert payload["memory_refs"][0]["memory_id"] == "memory-复用-1"


def test_append_across_logger_instances(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    TraceLogger(path).log_envelope(_envelope("msg-1", step_id="step-1"))
    TraceLogger(path).log_envelope(_envelope("msg-2", step_id="step-2"))

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert [json.loads(line)["message_id"] for line in lines] == ["msg-1", "msg-2"]


def test_unicode_and_deterministic_compact_json(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    TraceLogger(path).log_envelope(_envelope("msg-unicode", step_id="step-unicode"))

    line = path.read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(line)
    expected = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    assert "openEuler 网络恢复" in line
    assert "规划器-α" in line
    assert line == expected


def test_log_is_visible_before_return(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"

    event = TraceLogger(path).log_envelope(_envelope("msg-visible", step_id="step-visible"))

    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8"))["message_id"] == event.message_id


def test_invalid_logger_input() -> None:
    with pytest.raises(TypeError):
        TraceLogger(object())  # type: ignore[arg-type]


def test_log_envelope_requires_protocol_model(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        TraceLogger(tmp_path / "trace.jsonl").log_envelope({})  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), -float("inf"), True, "1"])
def test_recorded_at_must_be_finite_non_negative(value: object) -> None:
    payload = {
        "event_type": "message",
        "recorded_at": value,
        "trace_id": "trace",
        "task_id": "task",
        "message_id": "msg",
        "step_id": "step",
        "sender": "a",
        "receiver": "b",
        "message_type": "REQUEST",
        "action": None,
        "status": "OK",
        "state_refs": [],
        "artifact_refs": [],
        "memory_refs": [],
    }
    with pytest.raises(ValidationError):
        TraceEvent(**payload)


def test_trace_event_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(
            event_type="message",
            recorded_at=1.0,
            trace_id="trace",
            task_id="task",
            message_id="msg",
            step_id="step",
            sender="a",
            receiver="b",
            message_type="REQUEST",
            action=None,
            status="OK",
            state_refs=[],
            artifact_refs=[],
            memory_refs=[],
            extra=1,
        )
