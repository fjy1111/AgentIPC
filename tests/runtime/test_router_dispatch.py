import pytest

from agentipc.agents.base import BaseAgent
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.router import Router


class RecordingAgent(BaseAgent):
    capabilities = ["fake"]

    def __init__(
        self,
        agent_id: str,
        *,
        response: AgentEnvelope | None = None,
        error: Exception | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.response = response
        self.error = error
        self.call_count = 0
        self.received_envelope = None
        self.received_ctx = None

    def handle(self, envelope, ctx):
        self.call_count += 1
        self.received_envelope = envelope
        self.received_ctx = ctx
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


def _envelope(receiver: str = "fake", action: ActionType = ActionType.PLAN) -> AgentEnvelope:
    return AgentEnvelope(
        trace_id="trace-router",
        task_id="task-router",
        step_id="step-router",
        sender="runtime",
        receiver=receiver,
        message_type=MessageType.REQUEST,
        action=action,
    )


def _response() -> AgentEnvelope:
    return AgentEnvelope(
        trace_id="trace-router",
        task_id="task-router",
        step_id="step-router",
        sender="fake",
        receiver="runtime",
        message_type=MessageType.RESULT,
        action=ActionType.PLAN,
        status=MessageStatus.OK,
    )


def test_constructor_requires_agent_registry() -> None:
    with pytest.raises(TypeError):
        Router(object())  # type: ignore[arg-type]


def test_dispatch_preserves_input_and_result_identity() -> None:
    registry = AgentRegistry()
    response = _response()
    agent = RecordingAgent("fake", response=response)
    registry.register(agent)
    router = Router(registry)
    envelope = _envelope()
    ctx = object()

    result = router.dispatch(envelope, ctx)  # type: ignore[arg-type]

    assert agent.call_count == 1
    assert agent.received_envelope is envelope
    assert agent.received_ctx is ctx
    assert result is response


def test_receiver_selects_correct_agent_without_action_rerouting() -> None:
    registry = AgentRegistry()
    response_a = _response()
    response_b = _response()
    agent_a = RecordingAgent("agent-a", response=response_a)
    agent_b = RecordingAgent("agent-b", response=response_b)
    registry.register(agent_a)
    registry.register(agent_b)
    router = Router(registry)
    envelope = _envelope(receiver="agent-b", action=ActionType.RETRIEVE)

    result = router.dispatch(envelope, object())  # type: ignore[arg-type]

    assert agent_a.call_count == 0
    assert agent_b.call_count == 1
    assert result is response_b


def test_unknown_receiver_propagates_key_error_and_calls_no_agent() -> None:
    registry = AgentRegistry()
    agent = RecordingAgent("known", response=_response())
    registry.register(agent)
    router = Router(registry)

    with pytest.raises(KeyError):
        router.dispatch(_envelope(receiver="missing"), object())  # type: ignore[arg-type]

    assert agent.call_count == 0


@pytest.mark.parametrize("invalid_envelope", [None, {}, "envelope"])
def test_dispatch_rejects_non_envelope(invalid_envelope: object) -> None:
    router = Router(AgentRegistry())
    with pytest.raises(TypeError):
        router.dispatch(invalid_envelope, object())  # type: ignore[arg-type]


def test_agent_failure_is_propagated_unchanged() -> None:
    registry = AgentRegistry()
    error = RuntimeError("agent failure")
    agent = RecordingAgent("fake", error=error)
    registry.register(agent)
    router = Router(registry)

    with pytest.raises(RuntimeError, match="agent failure") as exc_info:
        router.dispatch(_envelope(), object())  # type: ignore[arg-type]

    assert exc_info.value is error
