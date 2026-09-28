from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from agentipc.agents.base import BaseAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.providers.mock_llm import MockLLMProvider


class PoisonService:
    def __getattr__(self, name: str):
        raise AssertionError(f"T085 must not access forbidden service: {name}")


class RecordingMockLLMProvider(MockLLMProvider):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.last_messages = None
        self.last_temperature = None

    def complete(self, messages, *, temperature=0.0):
        self.last_messages = deepcopy(messages)
        self.last_temperature = temperature
        return super().complete(messages, temperature=temperature)


class BadTextProvider:
    def __init__(self, text) -> None:
        self.text = text

    def complete(self, messages, *, temperature=0.0):
        return SimpleNamespace(text=self.text)


def _context(llm):
    poison = PoisonService()
    return SimpleNamespace(
        provider_bundle=SimpleNamespace(llm=llm, embedding=poison),
        state_hub=poison,
        memory_service=poison,
        artifact_store=poison,
        metrics=poison,
        trace_logger=poison,
    )


def _default_evidence():
    return [
        {
            "document_id": "network-manager",
            "text": "NetworkManager controls network connections.",
            "score": 0.95,
        },
        {
            "document_id": "network-check",
            "text": "Inspect active connections and routes.",
            "score": 0.80,
        },
    ]


def _default_execution():
    return {
        "operation": "identity",
        "output": {"status": "checked"},
    }


def _request(**overrides) -> AgentEnvelope:
    args = {
        "task": "diagnose openEuler network connectivity",
        "evidence": _default_evidence(),
        "execution": _default_execution(),
        "tags": ["network", "openEuler"],
        "keywords": ["NetworkManager", "connectivity"],
    }
    args.update(overrides)
    return AgentEnvelope(
        message_id="msg-summarize-request",
        trace_id="trace-summarize",
        task_id="task-summarize",
        step_id="step-summarize",
        sender="runtime",
        receiver="summarizer",
        message_type=MessageType.REQUEST,
        action=ActionType.SUMMARIZE,
        args=args,
        metrics={"sentinel": "unchanged"},
    )


def test_happy_path_is_exact_uses_provider_and_builds_memory_record() -> None:
    llm = RecordingMockLLMProvider(
        keyword_responses={
            "network": "Restart NetworkManager and verify connectivity.",
        }
    )
    request = _request()
    before = request.model_copy(deep=True)
    evidence_before = deepcopy(request.args["evidence"])
    execution_before = deepcopy(request.args["execution"])
    tags_before = deepcopy(request.args["tags"])
    keywords_before = deepcopy(request.args["keywords"])

    agent = SummarizerAgent()
    output = agent.handle(request, _context(llm))

    evidence_summary = (
        "network-manager: NetworkManager controls network connections. | "
        "network-check: Inspect active connections and routes."
    )
    answer = "Restart NetworkManager and verify connectivity."
    candidate = {
        "source_agent": "summarizer",
        "task_topic": "diagnose openEuler network connectivity",
        "summary": answer,
        "memory_type": "result",
        "tags": ["network", "openEuler"],
        "keywords": ["NetworkManager", "connectivity"],
        "payload": {
            "answer": answer,
            "evidence_summary": evidence_summary,
            "execution": _default_execution(),
        },
    }

    assert isinstance(agent, BaseAgent)
    assert SummarizerAgent.agent_id == "summarizer"
    assert SummarizerAgent.capabilities == ["summarize"]
    assert output.trace_id == request.trace_id
    assert output.task_id == request.task_id
    assert output.step_id == request.step_id
    assert output.sender == "summarizer"
    assert output.receiver == "runtime"
    assert output.message_type is MessageType.RESULT
    assert output.action is ActionType.SUMMARIZE
    assert output.status is MessageStatus.OK
    assert output.capability == "summarize"
    assert output.args == {}
    assert output.result == {
        "answer": answer,
        "evidence_summary": evidence_summary,
        "memory_candidate": candidate,
    }
    assert output.state_refs == []
    assert output.artifact_refs == []
    assert output.memory_refs == []

    assert llm.call_count == 1
    assert llm.last_temperature == 0.0
    expected_user = json.dumps(
        {
            "task": "diagnose openEuler network connectivity",
            "evidence_summary": evidence_summary,
            "execution": _default_execution(),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert llm.last_messages == [
        {
            "role": "system",
            "content": (
                "Produce a concise final answer for the AgentIPC task. "
                "Use only the supplied task, evidence summary, and execution result. "
                "Do not return protocol envelopes or hidden reasoning."
            ),
        },
        {"role": "user", "content": expected_user},
    ]

    record = MemoryRecord(memory_id="mem-test-result", **candidate)
    assert record.memory_type is MemoryType.RESULT
    assert record.source_agent == "summarizer"
    assert record.task_topic == "diagnose openEuler network connectivity"
    assert record.summary == answer

    assert request == before
    assert request.args["evidence"] == evidence_before
    assert request.args["execution"] == execution_before
    assert request.args["tags"] == tags_before
    assert request.args["keywords"] == keywords_before
    assert decode(encode(output)) == output


def test_empty_evidence_uses_fixed_summary_and_still_succeeds() -> None:
    llm = RecordingMockLLMProvider(default_text="No evidence answer")
    output = SummarizerAgent().handle(
        _request(evidence=[]),
        _context(llm),
    )

    assert output.result["answer"] == "No evidence answer"
    assert output.result["evidence_summary"] == "No retrieved evidence."
    assert output.result["memory_candidate"]["payload"]["evidence_summary"] == (
        "No retrieved evidence."
    )
    assert llm.call_count == 1


def test_only_first_three_evidence_items_are_summarized_without_mutation() -> None:
    evidence = [
        {"document_id": f"doc-{index}", "text": f"text-{index}", "score": index + 0.5}
        for index in range(5)
    ]
    before = deepcopy(evidence)
    output = SummarizerAgent().handle(
        _request(evidence=evidence),
        _context(MockLLMProvider(default_text="answer")),
    )

    assert output.result["evidence_summary"] == (
        "doc-0: text-0 | doc-1: text-1 | doc-2: text-2"
    )
    assert "doc-3" not in output.result["evidence_summary"]
    assert evidence == before


def test_optional_tags_and_keywords_default_to_empty_lists() -> None:
    request = _request()
    request.args.pop("tags")
    request.args.pop("keywords")
    output = SummarizerAgent().handle(
        request,
        _context(MockLLMProvider(default_text="answer")),
    )
    candidate = output.result["memory_candidate"]
    assert candidate["tags"] == []
    assert candidate["keywords"] == []


def test_tags_and_keywords_preserve_order_and_duplicates() -> None:
    output = SummarizerAgent().handle(
        _request(tags=["b", "a", "b"], keywords=["z", "y", "z"]),
        _context(MockLLMProvider(default_text="answer")),
    )
    candidate = output.result["memory_candidate"]
    assert candidate["tags"] == ["b", "a", "b"]
    assert candidate["keywords"] == ["z", "y", "z"]


def test_evidence_score_above_one_is_allowed() -> None:
    evidence = [{"document_id": "state-score", "text": "valid", "score": 1.25}]
    output = SummarizerAgent().handle(
        _request(evidence=evidence),
        _context(MockLLMProvider(default_text="answer")),
    )
    assert output.result["evidence_summary"] == "state-score: valid"


def test_rejects_wrong_message_type() -> None:
    request = _request().model_copy(update={"message_type": MessageType.RESULT})
    with pytest.raises(ValueError):
        SummarizerAgent().handle(request, _context(MockLLMProvider()))


def test_rejects_wrong_action() -> None:
    request = _request().model_copy(update={"action": ActionType.EXECUTE})
    with pytest.raises(ValueError):
        SummarizerAgent().handle(request, _context(MockLLMProvider()))


def test_rejects_missing_task() -> None:
    request = _request()
    request.args.pop("task")
    with pytest.raises(ValueError):
        SummarizerAgent().handle(request, _context(MockLLMProvider()))


@pytest.mark.parametrize("task", [None, 1, True, [], {}])
def test_task_must_be_exact_string(task: object) -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(_request(task=task), _context(MockLLMProvider()))


def test_task_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        SummarizerAgent().handle(_request(task=""), _context(MockLLMProvider()))


def test_rejects_missing_evidence() -> None:
    request = _request()
    request.args.pop("evidence")
    with pytest.raises(ValueError):
        SummarizerAgent().handle(request, _context(MockLLMProvider()))


@pytest.mark.parametrize("evidence", [None, (), {}, "evidence", 1, True])
def test_evidence_must_be_exact_list(evidence: object) -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(
            _request(evidence=evidence),
            _context(MockLLMProvider()),
        )


def test_evidence_item_must_be_exact_dict() -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(
            _request(evidence=["item"]),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize("missing", ["document_id", "text", "score"])
def test_evidence_item_requires_all_fields(missing: str) -> None:
    item = {"document_id": "doc", "text": "text", "score": 0.5}
    item.pop(missing)
    with pytest.raises(ValueError):
        SummarizerAgent().handle(
            _request(evidence=[item]),
            _context(MockLLMProvider()),
        )


def test_evidence_item_rejects_extra_fields() -> None:
    item = {"document_id": "doc", "text": "text", "score": 0.5, "extra": True}
    with pytest.raises(ValueError):
        SummarizerAgent().handle(
            _request(evidence=[item]),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("document_id", 1, TypeError),
        ("document_id", "", ValueError),
        ("text", 1, TypeError),
        ("text", "", ValueError),
        ("score", True, TypeError),
        ("score", "0.5", TypeError),
        ("score", -0.1, ValueError),
        ("score", float("nan"), ValueError),
        ("score", float("inf"), ValueError),
    ],
)
def test_evidence_item_field_validation(field, value, error) -> None:
    item = {"document_id": "doc", "text": "text", "score": 0.5}
    item[field] = value
    with pytest.raises(error):
        SummarizerAgent().handle(
            _request(evidence=[item]),
            _context(MockLLMProvider()),
        )


def test_rejects_missing_execution() -> None:
    request = _request()
    request.args.pop("execution")
    with pytest.raises(ValueError):
        SummarizerAgent().handle(request, _context(MockLLMProvider()))


@pytest.mark.parametrize("execution", [None, [], "execution", 1, True])
def test_execution_must_be_exact_dict(execution: object) -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(
            _request(execution=execution),
            _context(MockLLMProvider()),
        )


def test_execution_requires_operation() -> None:
    with pytest.raises(ValueError):
        SummarizerAgent().handle(
            _request(execution={"output": 1}),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize("operation", [None, 1, True, []])
def test_execution_operation_must_be_exact_string(operation: object) -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(
            _request(execution={"operation": operation}),
            _context(MockLLMProvider()),
        )


def test_execution_operation_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        SummarizerAgent().handle(
            _request(execution={"operation": ""}),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize(
    "execution",
    [
        {"operation": "identity", "output": object()},
        {"operation": "identity", "output": {1, 2}},
    ],
)
def test_execution_rejects_non_json_compatible_values(execution: dict) -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(
            _request(execution=execution),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_execution_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        SummarizerAgent().handle(
            _request(execution={"operation": "identity", "output": value}),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize("field", ["tags", "keywords"])
@pytest.mark.parametrize("value", [None, (), "x", 1, True])
def test_tags_and_keywords_must_be_exact_lists(field: str, value: object) -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(
            _request(**{field: value}),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize("field", ["tags", "keywords"])
def test_tags_and_keywords_reject_non_string_items(field: str) -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(
            _request(**{field: ["ok", 1]}),
            _context(MockLLMProvider()),
        )


@pytest.mark.parametrize("field", ["tags", "keywords"])
def test_tags_and_keywords_reject_empty_items(field: str) -> None:
    with pytest.raises(ValueError):
        SummarizerAgent().handle(
            _request(**{field: ["ok", ""]}),
            _context(MockLLMProvider()),
        )


def test_provider_empty_answer_is_rejected() -> None:
    llm = RecordingMockLLMProvider(default_text="")
    with pytest.raises(ValueError):
        SummarizerAgent().handle(_request(), _context(llm))
    assert llm.call_count == 1
    assert llm.last_temperature == 0.0


def test_provider_non_string_answer_is_rejected() -> None:
    with pytest.raises(TypeError):
        SummarizerAgent().handle(_request(), _context(BadTextProvider(123)))
