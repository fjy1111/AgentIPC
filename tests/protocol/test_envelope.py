import pytest
from pydantic import ValidationError

from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import (
    PROTOCOL_VERSION,
    ActionType,
    MessageStatus,
    MessageType,
)
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef


def make_envelope(**overrides) -> AgentEnvelope:
    data = {
        "step_id": "step_1",
        "sender": "planner",
        "receiver": "retriever",
        "message_type": MessageType.REQUEST,
    }
    data.update(overrides)
    return AgentEnvelope(**data)


def test_minimal_envelope_uses_expected_defaults() -> None:
    envelope = make_envelope()

    assert envelope.version == PROTOCOL_VERSION
    assert envelope.message_id
    assert envelope.trace_id
    assert envelope.task_id
    assert envelope.created_at > 0
    assert envelope.status is MessageStatus.PENDING
    assert envelope.state_refs == []
    assert envelope.artifact_refs == []
    assert envelope.memory_refs == []
    assert envelope.args == {}
    assert envelope.result is None
    assert envelope.metrics == {}
    assert envelope.action is None
    assert envelope.capability is None


def test_generated_ids_use_existing_helper_prefixes() -> None:
    envelope = make_envelope()

    assert envelope.message_id.startswith("msg_")
    assert envelope.trace_id.startswith("trace_")
    assert envelope.task_id.startswith("task_")


def test_mutable_defaults_are_independent_between_envelopes() -> None:
    first = make_envelope()
    second = make_envelope()

    first.args["query"] = "example"
    first.state_refs.append(
        StateRef(
            uri="shm://agentipc/state-1",
            kind="vector",
            shape=[2],
            dtype="float32",
            nbytes=8,
            checksum="checksum",
            transport="shm",
            summary="state",
        )
    )
    first.metrics["latency_ms"] = 1.5

    assert second.args == {}
    assert second.state_refs == []
    assert second.metrics == {}


def test_explicit_ids_are_preserved() -> None:
    envelope = make_envelope(
        message_id="msg_explicit",
        trace_id="trace_explicit",
        task_id="task_explicit",
    )

    assert envelope.message_id == "msg_explicit"
    assert envelope.trace_id == "trace_explicit"
    assert envelope.task_id == "task_explicit"


def test_reference_models_are_preserved_when_nested() -> None:
    state_ref = StateRef(
        uri="shm://agentipc/state-1",
        kind="vector",
        shape=[2],
        dtype="float32",
        nbytes=8,
        checksum="checksum",
        transport="shm",
        summary="state",
    )
    artifact_ref = ArtifactRef(
        uri="artifact://sha256/example",
        sha256="b" * 64,
        media_type="application/json",
        size_bytes=10,
        summary="artifact",
    )
    memory_ref = MemoryRef(
        memory_id="mem_1",
        score=0.8,
        match_type="hybrid",
        summary="memory",
    )

    envelope = make_envelope(
        state_refs=[state_ref],
        artifact_refs=[artifact_ref],
        memory_refs=[memory_ref],
    )

    assert isinstance(envelope.state_refs[0], StateRef)
    assert isinstance(envelope.artifact_refs[0], ArtifactRef)
    assert isinstance(envelope.memory_refs[0], MemoryRef)


def test_envelope_model_dump_and_validate_round_trip() -> None:
    envelope = make_envelope(
        action=ActionType.RETRIEVE,
        status=MessageStatus.OK,
        args={"query": "agent ipc"},
        result={"count": 2},
    )

    restored = AgentEnvelope.model_validate(envelope.model_dump())

    assert restored == envelope


def test_envelope_parses_enum_values_from_strings() -> None:
    envelope = make_envelope(
        message_type="REQUEST",
        action="PLAN",
        status="OK",
    )

    assert envelope.message_type is MessageType.REQUEST
    assert envelope.action is ActionType.PLAN
    assert envelope.status is MessageStatus.OK


def test_envelope_rejects_unknown_extra_field() -> None:
    with pytest.raises(ValidationError):
        make_envelope(unexpected=True)