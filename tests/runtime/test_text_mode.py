from pathlib import Path

import pytest

from agentipc.agents.base import BaseAgent
from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.text_counter import TextCounter
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.protocol.text_adapter import render as real_render
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.bootstrap import run_handshake
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router
from agentipc.runtime.text_transport import TextTransport


TASK = "diagnose openEuler network connectivity"

KNOWLEDGE = [
    {
        "document_id": "network-manager",
        "text": (
            "NetworkManager manages openEuler network "
            "connections and connectivity."
        ),
        "keywords": [
            "NetworkManager",
            "openEuler",
            "network connectivity",
        ],
    },
    {
        "document_id": "filesystem",
        "text": "Use fsck for filesystem diagnostics.",
        "keywords": [
            "filesystem",
            "fsck",
        ],
    },
]


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"forbidden infrastructure access: {name}")


class FailingPlanner(BaseAgent):
    agent_id = "planner"
    capabilities = ["plan"]

    def handle(self, envelope, ctx):
        raise RuntimeError("planner exploded")


def _events(path: Path) -> list[TraceEvent]:
    if not path.exists():
        return []
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _runtime(
    tmp_path: Path,
    *,
    mode: RunMode,
    name: str,
) -> tuple[
    AgentRegistry,
    CapabilityRegistry,
    MetricsCollector,
    TraceLogger,
    Path,
    RunContext,
]:
    agent_registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent(),
        SummarizerAgent(),
    ]:
        agent_registry.register(agent)

    capability_registry = CapabilityRegistry()
    metrics = MetricsCollector()
    trace_path = tmp_path / f"{name}.jsonl"
    trace_logger = TraceLogger(trace_path)
    provider_bundle = ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={
                "network": "Network troubleshooting completed.",
            },
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(),
    )
    poison = PoisonService()
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=mode,
        config=AgentIPCConfig(),
        registry=capability_registry,
        state_hub=poison,
        artifact_store=poison,
        memory_service=poison,
        metrics=metrics,
        trace_logger=trace_logger,
        provider_bundle=provider_bundle,
        use_state=False,
        use_memory=False,
        use_sandbox=False,
    )
    return (
        agent_registry,
        capability_registry,
        metrics,
        trace_logger,
        trace_path,
        ctx,
    )


def _business_events(path: Path) -> list[TraceEvent]:
    events = _events(path)
    return [
        event
        for event in events
        if event.message_type in {
            MessageType.REQUEST.value,
            MessageType.RESULT.value,
        }
    ]


def test_structured_and_text_modes_are_business_equivalent_and_metrics_isolated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        structured_registry,
        structured_capabilities,
        structured_metrics,
        structured_trace_logger,
        structured_trace_path,
        structured_ctx,
    ) = _runtime(
        tmp_path,
        mode=RunMode.STRUCTURED,
        name="structured",
    )
    (
        text_registry,
        text_capabilities,
        text_metrics,
        text_trace_logger,
        text_trace_path,
        text_ctx,
    ) = _runtime(
        tmp_path,
        mode=RunMode.TEXT,
        name="text",
    )

    structured_handshake = run_handshake(
        agent_registry=structured_registry,
        capability_registry=structured_capabilities,
        trace_logger=structured_trace_logger,
        trace_id=structured_ctx.trace_id,
        task_id=structured_ctx.task_id,
    )
    text_handshake = run_handshake(
        agent_registry=text_registry,
        capability_registry=text_capabilities,
        trace_logger=text_trace_logger,
        trace_id=text_ctx.trace_id,
        task_id=text_ctx.task_id,
    )
    assert len(structured_handshake) == 12
    assert len(text_handshake) == 12
    assert structured_metrics.snapshot().message_count == 0
    assert text_metrics.snapshot().message_count == 0

    rendered_messages: list[tuple[AgentEnvelope, str]] = []

    def recording_render(
        envelope: AgentEnvelope,
        resolver: object | None = None,
    ) -> str:
        before = envelope.model_copy(deep=True)
        text = real_render(
            envelope,
            resolver=resolver,
        )
        assert envelope == before
        rendered_messages.append((envelope, text))
        return text

    monkeypatch.setattr(
        "agentipc.runtime.text_transport.render",
        recording_render,
    )

    structured_final = Orchestrator(
        Router(structured_registry)
    ).run_task(
        task=TASK,
        ctx=structured_ctx,
    )
    text_final = Orchestrator(
        Router(text_registry),
        text_transport=TextTransport(
            text_counter=TextCounter(use_tiktoken=False),
        ),
    ).run_task(
        task=TASK,
        ctx=text_ctx,
    )

    assert structured_final.result == text_final.result
    assert structured_final.message_type is text_final.message_type
    assert structured_final.action is text_final.action
    assert structured_final.status is text_final.status
    assert structured_final.capability == text_final.capability
    assert structured_final.result is not None
    assert text_final.result is not None
    assert structured_final.result["answer"] == (
        "Network troubleshooting completed."
    )
    assert text_final.result["answer"] == "Network troubleshooting completed."
    assert "network-manager" in structured_final.result["evidence_summary"]
    assert "network-manager" in text_final.result["evidence_summary"]

    structured_execution = structured_final.result["memory_candidate"]["payload"][
        "execution"
    ]
    text_execution = text_final.result["memory_candidate"]["payload"]["execution"]
    assert structured_execution["operation"] == "identity"
    assert text_execution["operation"] == "identity"
    assert "network-manager" in structured_execution["output"][
        "retrieved_document_ids"
    ]
    assert "network-manager" in text_execution["output"][
        "retrieved_document_ids"
    ]

    assert len(rendered_messages) == 8
    assert [
        (envelope.message_type, envelope.action)
        for envelope, _ in rendered_messages
    ] == [
        (MessageType.REQUEST, ActionType.PLAN),
        (MessageType.RESULT, ActionType.PLAN),
        (MessageType.REQUEST, ActionType.RETRIEVE),
        (MessageType.RESULT, ActionType.RETRIEVE),
        (MessageType.REQUEST, ActionType.EXECUTE),
        (MessageType.RESULT, ActionType.EXECUTE),
        (MessageType.REQUEST, ActionType.SUMMARIZE),
        (MessageType.RESULT, ActionType.SUMMARIZE),
    ]
    assert all(envelope.metrics == {} for envelope, _ in rendered_messages)

    request_texts = {
        envelope.action: text
        for envelope, text in rendered_messages
        if envelope.message_type is MessageType.REQUEST
    }
    assert "Message type: REQUEST" in request_texts[ActionType.PLAN]
    assert "Action: PLAN" in request_texts[ActionType.PLAN]
    assert "To: planner" in request_texts[ActionType.PLAN]
    assert "Action: RETRIEVE" in request_texts[ActionType.RETRIEVE]
    assert "To: retriever" in request_texts[ActionType.RETRIEVE]
    assert "Action: EXECUTE" in request_texts[ActionType.EXECUTE]
    assert "To: executor" in request_texts[ActionType.EXECUTE]
    assert "Action: SUMMARIZE" in request_texts[ActionType.SUMMARIZE]
    assert "To: summarizer" in request_texts[ActionType.SUMMARIZE]

    structured_snapshot = structured_metrics.snapshot()
    text_snapshot = text_metrics.snapshot()
    expected_text_chars = sum(
        len(rendered_text)
        for _, rendered_text in rendered_messages
    )

    assert structured_snapshot.message_count == 8
    assert structured_snapshot.protocol_bytes > 0
    assert structured_snapshot.text_chars == 0
    assert structured_snapshot.text_tokens == 0

    assert text_snapshot.message_count == 8
    assert text_snapshot.protocol_bytes == 0
    assert text_snapshot.text_chars == expected_text_chars
    assert expected_text_chars > 0
    assert text_snapshot.text_tokens == 0

    structured_business = _business_events(structured_trace_path)
    text_business = _business_events(text_trace_path)
    assert len(structured_business) == 8
    assert len(text_business) == 8
    expected_sequence = [
        ("REQUEST", "PLAN"),
        ("RESULT", "PLAN"),
        ("REQUEST", "RETRIEVE"),
        ("RESULT", "RETRIEVE"),
        ("REQUEST", "EXECUTE"),
        ("RESULT", "EXECUTE"),
        ("REQUEST", "SUMMARIZE"),
        ("RESULT", "SUMMARIZE"),
    ]
    assert [
        (event.message_type, event.action)
        for event in structured_business
    ] == expected_sequence
    assert [
        (event.message_type, event.action)
        for event in text_business
    ] == expected_sequence

    structured_request_order = [
        event.action
        for event in structured_business
        if event.message_type == MessageType.REQUEST.value
    ]
    text_request_order = [
        event.action
        for event in text_business
        if event.message_type == MessageType.REQUEST.value
    ]
    assert structured_request_order == [
        "PLAN",
        "RETRIEVE",
        "EXECUTE",
        "SUMMARIZE",
    ]
    assert text_request_order == structured_request_order

    assert all(event.state_refs == [] for event in structured_business)
    assert all(event.artifact_refs == [] for event in structured_business)
    assert all(event.memory_refs == [] for event in structured_business)
    assert all(event.state_refs == [] for event in text_business)
    assert all(event.artifact_refs == [] for event in text_business)
    assert all(event.memory_refs == [] for event in text_business)


def test_failing_planner_text_mode_counts_only_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, metrics, _, trace_path, ctx = _runtime(
        tmp_path,
        mode=RunMode.TEXT,
        name="text-failure",
    )
    registry = AgentRegistry()
    registry.register(FailingPlanner())

    rendered_messages: list[tuple[AgentEnvelope, str]] = []

    def recording_render(
        envelope: AgentEnvelope,
        resolver: object | None = None,
    ) -> str:
        text = real_render(envelope, resolver=resolver)
        rendered_messages.append((envelope, text))
        return text

    monkeypatch.setattr(
        "agentipc.runtime.text_transport.render",
        recording_render,
    )

    orchestrator = Orchestrator(
        Router(registry),
        text_transport=TextTransport(
            text_counter=TextCounter(use_tiktoken=False),
        ),
    )
    with pytest.raises(RuntimeError, match="planner exploded"):
        orchestrator.run_task(task=TASK, ctx=ctx)

    snapshot = metrics.snapshot()
    assert len(rendered_messages) == 1
    request, rendered = rendered_messages[0]
    assert request.message_type is MessageType.REQUEST
    assert request.action is ActionType.PLAN
    assert snapshot.message_count == 1
    assert snapshot.protocol_bytes == 0
    assert snapshot.text_chars == len(rendered)
    assert snapshot.text_chars > 0
    assert snapshot.text_tokens == 0

    events = _events(trace_path)
    assert len(events) == 1
    assert events[0].message_type == MessageType.REQUEST.value
    assert events[0].action == ActionType.PLAN.value


def test_text_mode_without_transport_is_rejected_before_business_trace(
    tmp_path: Path,
) -> None:
    registry, _, metrics, _, trace_path, ctx = _runtime(
        tmp_path,
        mode=RunMode.TEXT,
        name="missing-transport",
    )

    with pytest.raises(ValueError, match="requires TextTransport"):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

    snapshot = metrics.snapshot()
    assert snapshot.message_count == 0
    assert snapshot.protocol_bytes == 0
    assert snapshot.text_chars == 0
    assert snapshot.text_tokens == 0
    assert _events(trace_path) == []


def test_structured_mode_does_not_require_text_transport(tmp_path: Path) -> None:
    registry, _, metrics, _, _, ctx = _runtime(
        tmp_path,
        mode=RunMode.STRUCTURED,
        name="structured-no-text-transport",
    )

    final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
    assert final.message_type is MessageType.RESULT
    assert final.action is ActionType.SUMMARIZE
    assert metrics.snapshot().message_count == 8


def test_text_transport_rejects_invalid_text_counter() -> None:
    with pytest.raises(TypeError, match="text_counter"):
        TextTransport(text_counter=object())  # type: ignore[arg-type]


def test_orchestrator_rejects_invalid_text_transport() -> None:
    with pytest.raises(TypeError, match="text_transport"):
        Orchestrator(
            Router(AgentRegistry()),
            text_transport=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [None, {}, "text"])
def test_text_transport_dispatch_rejects_invalid_envelope(
    tmp_path: Path,
    value: object,
) -> None:
    registry, _, _, _, trace_path, ctx = _runtime(
        tmp_path,
        mode=RunMode.TEXT,
        name=f"invalid-envelope-{type(value).__name__}",
    )
    transport = TextTransport(
        text_counter=TextCounter(use_tiktoken=False),
    )

    with pytest.raises(TypeError, match="envelope"):
        transport.dispatch(
            value,  # type: ignore[arg-type]
            ctx=ctx,
            router=Router(registry),
        )
    assert _events(trace_path) == []


def test_text_transport_dispatch_rejects_invalid_router(tmp_path: Path) -> None:
    _, _, _, _, trace_path, ctx = _runtime(
        tmp_path,
        mode=RunMode.TEXT,
        name="invalid-router",
    )
    envelope = AgentEnvelope(
        trace_id=ctx.trace_id,
        task_id=ctx.task_id,
        step_id="step-plan",
        sender="runtime",
        receiver="planner",
        message_type=MessageType.REQUEST,
        action=ActionType.PLAN,
        args={"task": TASK},
    )

    with pytest.raises(TypeError, match="router"):
        TextTransport(
            text_counter=TextCounter(use_tiktoken=False),
        ).dispatch(
            envelope,
            ctx=ctx,
            router=object(),  # type: ignore[arg-type]
        )
    assert _events(trace_path) == []


def test_render_still_occurs_when_metrics_is_not_a_collector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, _, _, trace_path, ctx = _runtime(
        tmp_path,
        mode=RunMode.TEXT,
        name="mock-metrics",
    )
    ctx.metrics = object()  # type: ignore[assignment]
    registry = AgentRegistry()
    registry.register(FailingPlanner())

    rendered: list[str] = []

    def recording_render(
        envelope: AgentEnvelope,
        resolver: object | None = None,
    ) -> str:
        text = real_render(envelope, resolver=resolver)
        rendered.append(text)
        return text

    monkeypatch.setattr(
        "agentipc.runtime.text_transport.render",
        recording_render,
    )

    with pytest.raises(RuntimeError, match="planner exploded"):
        Orchestrator(
            Router(registry),
            text_transport=TextTransport(
                text_counter=TextCounter(use_tiktoken=False),
            ),
        ).run_task(task=TASK, ctx=ctx)

    assert len(rendered) == 1
    assert "Message type: REQUEST" in rendered[0]
    assert len(_events(trace_path)) == 1
