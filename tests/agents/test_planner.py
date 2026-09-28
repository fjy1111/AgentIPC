from types import SimpleNamespace

import pytest

from agentipc.agents.base import BaseAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.providers.mock_llm import MockLLMProvider


class ForbiddenService:
    def __getattr__(self, name: str):
        raise AssertionError(f"T081 must not access infrastructure service: {name}")


class RecordingMockLLMProvider(MockLLMProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.last_messages = None
        self.last_temperature = None

    def complete(self, messages, *, temperature=0.0):
        self.last_messages = messages
        self.last_temperature = temperature
        return super().complete(messages, temperature=temperature)


def _context(llm: MockLLMProvider):
    return SimpleNamespace(
        provider_bundle=SimpleNamespace(llm=llm),
        state_hub=ForbiddenService(),
        memory_service=ForbiddenService(),
        artifact_store=ForbiddenService(),
    )


def _request(**args_overrides) -> AgentEnvelope:
    args = {
        "task": "diagnose openEuler network connectivity",
        "retrieval_topics": [
            "NetworkManager",
            "network connectivity",
        ],
        "requires_execution": False,
    }
    args.update(args_overrides)
    return AgentEnvelope(
        message_id="msg-plan-request",
        trace_id="trace-plan",
        task_id="task-plan",
        step_id="step-plan",
        sender="runtime",
        receiver="planner",
        message_type=MessageType.REQUEST,
        action=ActionType.PLAN,
        args=args,
        metrics={"sentinel": "unchanged"},
    )


def test_planner_happy_path_is_stable_and_uses_provider() -> None:
    llm = RecordingMockLLMProvider(
        keyword_responses={
            "network": "inspect network configuration first",
        }
    )
    planner = PlannerAgent()
    ctx = _context(llm)
    input_envelope = _request()
    before = input_envelope.model_copy(deep=True)

    output = planner.handle(input_envelope, ctx)

    assert isinstance(planner, BaseAgent)
    assert PlannerAgent.agent_id == "planner"
    assert PlannerAgent.capabilities == ["plan"]
    assert isinstance(output, AgentEnvelope)
    assert output is not input_envelope
    assert output.trace_id == input_envelope.trace_id
    assert output.task_id == input_envelope.task_id
    assert output.step_id == input_envelope.step_id
    assert output.sender == "planner"
    assert output.receiver == "runtime"
    assert output.message_type is MessageType.RESULT
    assert output.action is ActionType.PLAN
    assert output.status is MessageStatus.OK
    assert output.capability == "plan"
    assert output.args == {}
    assert output.state_refs == []
    assert output.artifact_refs == []
    assert output.memory_refs == []
    assert output.result == {
        "plan": {
            "task": "diagnose openEuler network connectivity",
            "steps": [
                "retrieve",
                "execute",
                "summarize",
            ],
            "required_capabilities": [
                "retrieve",
                "execute",
                "summarize",
            ],
            "retrieval_topics": [
                "NetworkManager",
                "network connectivity",
            ],
            "requires_execution": False,
            "planner_note": "inspect network configuration first",
        }
    }
    assert llm.call_count == 1
    assert llm.last_temperature == 0.0
    assert llm.last_messages == [
        {
            "role": "system",
            "content": (
                "Produce a concise planning note for the AgentIPC planner. "
                "Do not return protocol envelopes."
            ),
        },
        {
            "role": "user",
            "content": "diagnose openEuler network connectivity",
        },
    ]
    assert input_envelope == before
    assert decode(encode(output)) == output


def test_planner_defaults_optional_hints() -> None:
    llm = MockLLMProvider(default_text="note")
    planner = PlannerAgent()
    envelope = AgentEnvelope(
        message_id="msg-default",
        trace_id="trace-default",
        task_id="task-default",
        step_id="step-default",
        sender="runtime",
        receiver="planner",
        message_type=MessageType.REQUEST,
        action=ActionType.PLAN,
        args={"task": "summarize kernel logs"},
    )

    output = planner.handle(envelope, _context(llm))
    plan = output.result["plan"]  # type: ignore[index]

    assert plan["retrieval_topics"] == ["summarize kernel logs"]
    assert plan["requires_execution"] is False
    assert plan["planner_note"] == "note"
    assert llm.call_count == 1


def test_planner_rejects_missing_task() -> None:
    planner = PlannerAgent()
    envelope = _request()
    envelope.args.pop("task")

    with pytest.raises(ValueError):
        planner.handle(envelope, _context(MockLLMProvider()))


@pytest.mark.parametrize("task", [123, None, True])
def test_planner_rejects_non_string_task(task: object) -> None:
    with pytest.raises(TypeError):
        PlannerAgent().handle(
            _request(task=task),
            _context(MockLLMProvider()),
        )


def test_planner_rejects_empty_task() -> None:
    with pytest.raises(ValueError):
        PlannerAgent().handle(
            _request(task=""),
            _context(MockLLMProvider()),
        )


def test_planner_rejects_non_request_message_type() -> None:
    envelope = _request().model_copy(update={"message_type": MessageType.RESULT})

    with pytest.raises(ValueError):
        PlannerAgent().handle(envelope, _context(MockLLMProvider()))


def test_planner_rejects_non_plan_action() -> None:
    envelope = _request().model_copy(update={"action": ActionType.RETRIEVE})

    with pytest.raises(ValueError):
        PlannerAgent().handle(envelope, _context(MockLLMProvider()))


@pytest.mark.parametrize("topics", ["network", ("network",), None, 1])
def test_planner_rejects_retrieval_topics_that_are_not_list(topics: object) -> None:
    with pytest.raises(TypeError):
        PlannerAgent().handle(
            _request(retrieval_topics=topics),
            _context(MockLLMProvider()),
        )


def test_planner_rejects_non_string_retrieval_topic() -> None:
    with pytest.raises(TypeError):
        PlannerAgent().handle(
            _request(retrieval_topics=["network", 1]),
            _context(MockLLMProvider()),
        )


def test_planner_rejects_empty_retrieval_topic() -> None:
    with pytest.raises(ValueError):
        PlannerAgent().handle(
            _request(retrieval_topics=["network", ""]),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize("requires_execution", [1, 0, "false", None])
def test_planner_requires_exact_bool(requires_execution: object) -> None:
    with pytest.raises(TypeError):
        PlannerAgent().handle(
            _request(requires_execution=requires_execution),
            _context(MockLLMProvider()),
        )
