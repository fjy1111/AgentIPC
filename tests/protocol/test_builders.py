import json

import pytest

from agentipc.protocol.builders import (
    build_ack,
    build_discover,
    build_hello,
    build_register,
)
from agentipc.protocol.capability import AgentCapability
from agentipc.protocol.enums import (
    PROTOCOL_VERSION,
    MessageStatus,
    MessageType,
)


COMMON = {
    "sender": "planner",
    "receiver": "runtime",
    "trace_id": "trace_control",
    "task_id": "task_control",
}


def test_build_hello_creates_expected_envelope() -> None:
    envelope = build_hello(
        **COMMON,
        step_id="step_hello",
    )

    assert envelope.message_type is MessageType.HELLO
    assert envelope.sender == "planner"
    assert envelope.receiver == "runtime"
    assert envelope.trace_id == "trace_control"
    assert envelope.task_id == "task_control"
    assert envelope.step_id == "step_hello"
    assert envelope.args == {}
    assert envelope.action is None
    assert envelope.capability is None
    assert envelope.result is None
    assert envelope.status is MessageStatus.PENDING
    assert envelope.state_refs == []
    assert envelope.artifact_refs == []
    assert envelope.memory_refs == []


def test_build_register_embeds_json_compatible_capability_dict() -> None:
    capability = AgentCapability(
        agent_id="planner",
        capabilities=["planning"],
        metadata={"role": "planner"},
    )

    envelope = build_register(
        **COMMON,
        step_id="step_register",
        capability=capability,
    )

    assert envelope.message_type is MessageType.REGISTER
    assert envelope.action is None
    assert envelope.capability is None
    assert envelope.status is MessageStatus.PENDING
    assert envelope.args["capability"] == {
        "agent_id": "planner",
        "capabilities": ["planning"],
        "protocol_versions": [PROTOCOL_VERSION],
        "metadata": {"role": "planner"},
    }
    assert isinstance(envelope.args["capability"], dict)

    json.dumps(envelope.args)


def test_build_discover_uses_envelope_capability_field() -> None:
    envelope = build_discover(
        **COMMON,
        step_id="step_discover",
        required="retrieval",
    )

    assert envelope.message_type is MessageType.DISCOVER
    assert envelope.capability == "retrieval"
    assert envelope.args == {}
    assert envelope.action is None
    assert envelope.status is MessageStatus.PENDING


def test_build_discover_rejects_empty_required() -> None:
    with pytest.raises(ValueError, match="required"):
        build_discover(
            **COMMON,
            step_id="step_discover",
            required="",
        )


def test_build_ack_creates_ok_envelope_for_original_message() -> None:
    envelope = build_ack(
        **COMMON,
        step_id="step_ack",
        ack_message_id="msg_original",
    )

    assert envelope.message_type is MessageType.ACK
    assert envelope.action is None
    assert envelope.capability is None
    assert envelope.status is MessageStatus.OK
    assert envelope.args == {
        "ack_message_id": "msg_original",
    }


def test_build_ack_rejects_empty_ack_message_id() -> None:
    with pytest.raises(ValueError, match="ack_message_id"):
        build_ack(
            **COMMON,
            step_id="step_ack",
            ack_message_id="",
        )


def test_control_messages_get_independent_message_ids() -> None:
    first = build_hello(
        **COMMON,
        step_id="step_1",
    )
    second = build_hello(
        **COMMON,
        step_id="step_2",
    )

    assert first.message_id != second.message_id


def test_all_builders_preserve_trace_and_task_ids() -> None:
    capability = AgentCapability(
        agent_id="planner",
        capabilities=["planning"],
    )

    envelopes = [
        build_hello(
            **COMMON,
            step_id="step_hello",
        ),
        build_register(
            **COMMON,
            step_id="step_register",
            capability=capability,
        ),
        build_discover(
            **COMMON,
            step_id="step_discover",
            required="retrieval",
        ),
        build_ack(
            **COMMON,
            step_id="step_ack",
            ack_message_id="msg_original",
        ),
    ]

    for envelope in envelopes:
        assert envelope.trace_id == "trace_control"
        assert envelope.task_id == "task_control"