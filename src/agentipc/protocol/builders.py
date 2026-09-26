from agentipc.protocol.capability import AgentCapability
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import MessageStatus, MessageType


def build_hello(
    *,
    sender: str,
    receiver: str,
    trace_id: str,
    task_id: str,
    step_id: str,
) -> AgentEnvelope:
    return AgentEnvelope(
        sender=sender,
        receiver=receiver,
        trace_id=trace_id,
        task_id=task_id,
        step_id=step_id,
        message_type=MessageType.HELLO,
        action=None,
        capability=None,
        args={},
        result=None,
        status=MessageStatus.PENDING,
    )


def build_register(
    *,
    sender: str,
    receiver: str,
    trace_id: str,
    task_id: str,
    step_id: str,
    capability: AgentCapability,
) -> AgentEnvelope:
    return AgentEnvelope(
        sender=sender,
        receiver=receiver,
        trace_id=trace_id,
        task_id=task_id,
        step_id=step_id,
        message_type=MessageType.REGISTER,
        action=None,
        capability=None,
        args={
            "capability": capability.model_dump(mode="json"),
        },
        result=None,
        status=MessageStatus.PENDING,
    )


def build_discover(
    *,
    sender: str,
    receiver: str,
    trace_id: str,
    task_id: str,
    step_id: str,
    required: str,
) -> AgentEnvelope:
    if not required:
        raise ValueError("required must be a non-empty string")

    return AgentEnvelope(
        sender=sender,
        receiver=receiver,
        trace_id=trace_id,
        task_id=task_id,
        step_id=step_id,
        message_type=MessageType.DISCOVER,
        action=None,
        capability=required,
        args={},
        result=None,
        status=MessageStatus.PENDING,
    )


def build_ack(
    *,
    sender: str,
    receiver: str,
    trace_id: str,
    task_id: str,
    step_id: str,
    ack_message_id: str,
) -> AgentEnvelope:
    if not ack_message_id:
        raise ValueError("ack_message_id must be a non-empty string")

    return AgentEnvelope(
        sender=sender,
        receiver=receiver,
        trace_id=trace_id,
        task_id=task_id,
        step_id=step_id,
        message_type=MessageType.ACK,
        action=None,
        capability=None,
        args={
            "ack_message_id": ack_message_id,
        },
        result=None,
        status=MessageStatus.OK,
    )