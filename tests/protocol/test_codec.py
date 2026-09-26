import json

import pytest
from pydantic import ValidationError

from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef


def make_envelope(**overrides) -> AgentEnvelope:
    data = {
        "message_id": "msg_codec",
        "trace_id": "trace_codec",
        "task_id": "task_codec",
        "step_id": "step_1",
        "sender": "planner",
        "receiver": "retriever",
        "message_type": MessageType.REQUEST,
        "action": ActionType.RETRIEVE,
        "capability": "retrieval",
        "args": {"query": "agent ipc"},
        "result": {"count": 2},
        "status": MessageStatus.OK,
        "created_at": 1_700_000_000.25,
        "metrics": {"latency_ms": 1.5},
    }
    data.update(overrides)
    return AgentEnvelope(**data)


def test_encode_decode_round_trip() -> None:
    original = make_envelope()

    restored = decode(encode(original))

    assert restored == original


def test_encode_returns_bytes_without_mutating_envelope() -> None:
    envelope = make_envelope()
    before = envelope.model_copy(deep=True)

    payload = encode(envelope)

    assert isinstance(payload, bytes)
    assert envelope == before


def test_unicode_round_trip_uses_utf8_without_ascii_escaping() -> None:
    envelope = make_envelope(
        args={"query": "openEuler 服务状态"},
        result={"summary": "多智能体通信"},
    )

    payload = encode(envelope)
    restored = decode(payload)

    assert "openEuler 服务状态".encode("utf-8") in payload
    assert "多智能体通信".encode("utf-8") in payload
    assert restored.args == envelope.args
    assert restored.result == envelope.result


def test_nested_refs_round_trip_as_models() -> None:
    envelope = make_envelope(
        state_refs=[
            StateRef(
                uri="shm://agentipc/state-1",
                kind="plan_embedding",
                shape=[2, 4],
                dtype="float32",
                nbytes=32,
                checksum="state-checksum",
                transport="shm",
                summary="planner state",
            )
        ],
        artifact_refs=[
            ArtifactRef(
                uri="artifact://sha256/example",
                sha256="a" * 64,
                media_type="application/json",
                size_bytes=42,
                summary="retrieval evidence",
            )
        ],
        memory_refs=[
            MemoryRef(
                memory_id="mem_1",
                score=0.9,
                match_type="hybrid",
                summary="reusable result",
            )
        ],
    )

    restored = decode(encode(envelope))

    assert isinstance(restored.state_refs[0], StateRef)
    assert isinstance(restored.artifact_refs[0], ArtifactRef)
    assert isinstance(restored.memory_refs[0], MemoryRef)
    assert restored == envelope


def test_enums_round_trip_as_enum_members() -> None:
    envelope = make_envelope(
        message_type=MessageType.RESULT,
        action=ActionType.SUMMARIZE,
        status=MessageStatus.PENDING,
    )

    restored = decode(encode(envelope))

    assert restored.message_type is MessageType.RESULT
    assert restored.action is ActionType.SUMMARIZE
    assert restored.status is MessageStatus.PENDING


def test_encode_produces_compact_json() -> None:
    payload = encode(make_envelope())

    assert b"\n" not in payload
    assert b": " not in payload
    assert b", " not in payload
    assert json.loads(payload.decode("utf-8"))


def test_encode_is_deterministic() -> None:
    envelope = make_envelope(
        args={"z": 1, "a": 2},
        metrics={"z_metric": 3, "a_metric": 4},
    )

    assert encode(envelope) == encode(envelope)


def test_decode_rejects_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        decode(b"{not-json")


def test_decode_rejects_invalid_envelope_schema() -> None:
    payload = b'{"version":"agentipc/0.1"}'

    with pytest.raises(ValidationError):
        decode(payload)