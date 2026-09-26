from agentipc.protocol.builders import (
    build_ack,
    build_discover,
    build_hello,
    build_register,
)
from agentipc.protocol.capability import AgentCapability
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.enums import MessageStatus, MessageType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.protocol.text_adapter import render


TRACE_ID = "trace_protocol_integration"
TASK_ID = "task_protocol_integration"


def test_control_protocol_registration_and_discovery_flow() -> None:
    registry = CapabilityRegistry()
    retriever = AgentCapability(
        agent_id="retriever",
        capabilities=["retrieval", "search"],
    )

    register = build_register(
        sender="retriever",
        receiver="runtime",
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step_register",
        capability=retriever,
    )
    decoded_register = decode(encode(register))

    assert decoded_register == register
    assert decoded_register.message_type is MessageType.REGISTER

    restored_capability = AgentCapability.model_validate(
        decoded_register.args["capability"]
    )
    assert restored_capability == retriever

    registry.register(restored_capability)

    assert registry.get("retriever") == restored_capability
    assert registry.supports("retriever", "retrieval") is True
    assert registry.supports("retriever", "planning") is False

    discover = build_discover(
        sender="planner",
        receiver="runtime",
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step_discover",
        required="retrieval",
    )
    decoded_discover = decode(encode(discover))

    assert decoded_discover == discover
    assert decoded_discover.message_type is MessageType.DISCOVER

    required = decoded_discover.capability
    assert required == "retrieval"
    assert isinstance(required, str)

    matches = registry.discover(required)

    assert matches == [restored_capability]
    assert len(matches) == 1
    assert matches[0].agent_id == "retriever"
    assert isinstance(matches[0], AgentCapability)


def test_control_messages_support_structured_and_text_representations() -> None:
    hello = build_hello(
        sender="planner",
        receiver="runtime",
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step_hello",
    )
    hello_payload = encode(hello)
    decoded_hello = decode(hello_payload)
    hello_text = render(decoded_hello)

    assert decoded_hello == hello
    assert isinstance(hello_payload, bytes)
    assert isinstance(hello_text, str)
    for fragment in (
        "HELLO",
        "planner",
        "runtime",
        TRACE_ID,
    ):
        assert fragment in hello_text

    retriever = AgentCapability(
        agent_id="retriever",
        capabilities=["retrieval", "search"],
    )
    register = build_register(
        sender="retriever",
        receiver="runtime",
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step_register",
        capability=retriever,
    )
    discover = build_discover(
        sender="planner",
        receiver="runtime",
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step_discover",
        required="retrieval",
    )

    register_payload = encode(register)
    register_text = render(register)
    decoded_register = decode(register_payload)

    assert isinstance(register_payload, bytes)
    assert isinstance(register_text, str)
    assert decoded_register.message_type is MessageType.REGISTER
    assert decoded_register.sender == "retriever"
    assert decoded_register.receiver == "runtime"
    for fragment in (
        "REGISTER",
        "retriever",
        "runtime",
        "retrieval",
        "search",
    ):
        assert fragment in register_text

    discover_payload = encode(discover)
    discover_text = render(discover)
    decoded_discover = decode(discover_payload)

    assert isinstance(discover_payload, bytes)
    assert isinstance(discover_text, str)
    assert decoded_discover.message_type is MessageType.DISCOVER
    assert decoded_discover.capability == "retrieval"
    assert decoded_discover.sender == "planner"
    assert decoded_discover.receiver == "runtime"
    for fragment in (
        "DISCOVER",
        "retrieval",
        "planner",
        "runtime",
    ):
        assert fragment in discover_text

    ack = build_ack(
        sender="runtime",
        receiver="planner",
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step_ack",
        ack_message_id=decoded_discover.message_id,
    )
    decoded_ack = decode(encode(ack))
    ack_text = render(decoded_ack)

    assert decoded_ack == ack
    assert decoded_ack.message_type is MessageType.ACK
    assert decoded_ack.status is MessageStatus.OK
    assert decoded_ack.args["ack_message_id"] == decoded_discover.message_id
    assert isinstance(ack_text, str)
    for fragment in (
        "ACK",
        "runtime",
        "planner",
        decoded_discover.message_id,
    ):
        assert fragment in ack_text


def test_control_flow_preserves_task_identity_with_unique_messages() -> None:
    retriever = AgentCapability(
        agent_id="retriever",
        capabilities=["retrieval", "search"],
    )

    hello = decode(
        encode(
            build_hello(
                sender="planner",
                receiver="runtime",
                trace_id=TRACE_ID,
                task_id=TASK_ID,
                step_id="step_hello",
            )
        )
    )
    register = decode(
        encode(
            build_register(
                sender="retriever",
                receiver="runtime",
                trace_id=TRACE_ID,
                task_id=TASK_ID,
                step_id="step_register",
                capability=retriever,
            )
        )
    )
    discover = decode(
        encode(
            build_discover(
                sender="planner",
                receiver="runtime",
                trace_id=TRACE_ID,
                task_id=TASK_ID,
                step_id="step_discover",
                required="retrieval",
            )
        )
    )
    ack = decode(
        encode(
            build_ack(
                sender="runtime",
                receiver="planner",
                trace_id=TRACE_ID,
                task_id=TASK_ID,
                step_id="step_ack",
                ack_message_id=discover.message_id,
            )
        )
    )

    messages = [hello, register, discover, ack]

    assert {message.trace_id for message in messages} == {TRACE_ID}
    assert {message.task_id for message in messages} == {TASK_ID}
    assert len({message.message_id for message in messages}) == 4
    assert [message.message_type for message in messages] == [
        MessageType.HELLO,
        MessageType.REGISTER,
        MessageType.DISCOVER,
        MessageType.ACK,
    ]

    for message in messages:
        assert message.state_refs == []
        assert message.artifact_refs == []
        assert message.memory_refs == []