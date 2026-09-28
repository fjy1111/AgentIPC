import pytest

from agentipc.agents.base import BaseAgent
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import MessageType


def _envelope() -> AgentEnvelope:
    return AgentEnvelope(
        message_id="msg-base",
        trace_id="trace-base",
        task_id="task-base",
        step_id="step-base",
        sender="runtime",
        receiver="agent",
        message_type=MessageType.REQUEST,
    )


def test_base_agent_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAgent()  # type: ignore[abstract]


def test_subclass_without_handle_is_abstract() -> None:
    class MissingHandleAgent(BaseAgent):
        agent_id = "missing"
        capabilities = []

    with pytest.raises(TypeError):
        MissingHandleAgent()  # type: ignore[abstract]


def test_concrete_subclass_can_be_instantiated_and_handles_envelope() -> None:
    class ConcreteAgent(BaseAgent):
        agent_id = "concrete"
        capabilities = ["test"]

        def handle(self, envelope, ctx):
            return envelope

    agent = ConcreteAgent()
    envelope = _envelope()

    assert agent.agent_id == "concrete"
    assert agent.capabilities == ["test"]
    assert agent.handle(envelope, object()) is envelope
