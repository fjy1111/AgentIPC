from pathlib import Path

import pytest

from agentipc.agents.base import BaseAgent
from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.protocol.codec import encode as real_encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.bootstrap import run_handshake
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router


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


def _events(path: Path) -> list[TraceEvent]:
    if not path.exists():
        return []
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _runtime(tmp_path: Path) -> tuple[
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
    trace_path = tmp_path / "trace.jsonl"
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
        trace_id="trace-structured-metrics",
        task_id="task-structured-metrics",
        mode=RunMode.STRUCTURED,
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


def test_successful_structured_task_records_exact_transport_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        agent_registry,
        capability_registry,
        metrics,
        trace_logger,
        trace_path,
        ctx,
    ) = _runtime(tmp_path)

    encoded: list[tuple[AgentEnvelope, bytes]] = []

    def recording_encode(envelope: AgentEnvelope) -> bytes:
        before = envelope.model_copy(deep=True)
        payload = real_encode(envelope)
        assert envelope == before
        encoded.append((envelope, payload))
        return payload

    monkeypatch.setattr(
        "agentipc.runtime.orchestrator.encode",
        recording_encode,
    )

    handshake_messages = run_handshake(
        agent_registry=agent_registry,
        capability_registry=capability_registry,
        trace_logger=trace_logger,
        trace_id=ctx.trace_id,
        task_id=ctx.task_id,
    )
    assert len(handshake_messages) == 12

    before = metrics.snapshot()
    assert before.message_count == 0
    assert before.protocol_bytes == 0

    final = Orchestrator(Router(agent_registry)).run_task(task=TASK, ctx=ctx)

    assert final.result is not None
    assert final.result["answer"] == "Network troubleshooting completed."

    snapshot = metrics.snapshot()
    expected_protocol_bytes = sum(len(payload) for _, payload in encoded)

    assert len(encoded) == 8
    assert snapshot.message_count == 8
    assert expected_protocol_bytes > 0
    assert snapshot.protocol_bytes == expected_protocol_bytes

    assert [
        (envelope.message_type, envelope.action)
        for envelope, _ in encoded
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
    assert all(envelope.metrics == {} for envelope, _ in encoded)

    events = _events(trace_path)
    assert len(events) == 20
    business_events = events[-8:]
    assert len(business_events) == snapshot.message_count
    assert [event.message_id for event in business_events] == [
        envelope.message_id
        for envelope, _ in encoded
    ]

    assert snapshot.text_chars == 0
    assert snapshot.text_tokens == 0
    assert snapshot.state_transfer_count == 0
    assert snapshot.state_bytes == 0
    assert snapshot.artifact_ref_count == 0
    assert snapshot.memory_retrieved == 0
    assert snapshot.memory_used == 0
    assert snapshot.memory_effective == 0
    assert snapshot.memory_harmful == 0
    assert snapshot.tool_call_count == 0
    assert snapshot.repeated_tool_call_count == 0
    assert snapshot.latency_ms == 0.0
    assert snapshot.success is False


def test_transport_metrics_accumulate_without_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_registry, _, metrics, _, _, ctx = _runtime(tmp_path)
    metrics.increment("message_count", 2)
    metrics.increment("protocol_bytes", 100)

    encoded_payloads: list[bytes] = []

    def recording_encode(envelope: AgentEnvelope) -> bytes:
        payload = real_encode(envelope)
        encoded_payloads.append(payload)
        return payload

    monkeypatch.setattr(
        "agentipc.runtime.orchestrator.encode",
        recording_encode,
    )

    Orchestrator(Router(agent_registry)).run_task(task=TASK, ctx=ctx)

    snapshot = metrics.snapshot()
    assert len(encoded_payloads) == 8
    assert snapshot.message_count == 10
    assert snapshot.protocol_bytes == 100 + sum(
        len(payload)
        for payload in encoded_payloads
    )


class FailingPlanner(BaseAgent):
    agent_id = "planner"
    capabilities = ["plan"]

    def handle(self, envelope, ctx):
        raise RuntimeError("planner exploded")


def test_failing_planner_counts_only_sent_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = AgentRegistry()
    registry.register(FailingPlanner())
    _, _, metrics, _, trace_path, ctx = _runtime(tmp_path)

    encoded: list[tuple[AgentEnvelope, bytes]] = []

    def recording_encode(envelope: AgentEnvelope) -> bytes:
        payload = real_encode(envelope)
        encoded.append((envelope, payload))
        return payload

    monkeypatch.setattr(
        "agentipc.runtime.orchestrator.encode",
        recording_encode,
    )

    with pytest.raises(RuntimeError, match="planner exploded"):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

    snapshot = metrics.snapshot()
    assert len(encoded) == 1
    assert encoded[0][0].message_type is MessageType.REQUEST
    assert encoded[0][0].action is ActionType.PLAN
    assert snapshot.message_count == 1
    assert snapshot.protocol_bytes == len(encoded[0][1])

    events = _events(trace_path)
    assert len(events) == 1
    assert events[0].message_id == encoded[0][0].message_id


def test_pre_dispatch_failure_records_zero_transport_metrics(tmp_path: Path) -> None:
    agent_registry, _, metrics, _, trace_path, ctx = _runtime(tmp_path)

    with pytest.raises(ValueError, match="non-empty"):
        Orchestrator(Router(agent_registry)).run_task(task="", ctx=ctx)

    snapshot = metrics.snapshot()
    assert snapshot.message_count == 0
    assert snapshot.protocol_bytes == 0
    assert _events(trace_path) == []
