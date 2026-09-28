from __future__ import annotations

from pathlib import Path
import re

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.text_counter import TextCounter
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.codec import encode as real_encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.protocol.text_adapter import render as real_render
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.bootstrap import run_handshake
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.reference_resolver import ReferenceResolver
from agentipc.runtime.router import Router
from agentipc.runtime.text_transport import TextTransport
from agentipc.state.hub import StateHub


TASK = "diagnose openEuler network connectivity"
EVIDENCE_SENTINEL = (
    "NetworkManager manages openEuler network connections and connectivity."
)

KNOWLEDGE = [
    {
        "document_id": "network-manager",
        "text": EVIDENCE_SENTINEL,
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


class RecordingArtifactStore(ArtifactStore):
    def __init__(self, root: str | Path) -> None:
        super().__init__(root)
        self.put_json_calls = 0
        self.get_json_calls = 0

    def put_json(self, value: object, *, summary: str):
        self.put_json_calls += 1
        return super().put_json(value, summary=summary)

    def get_json(self, ref):
        self.get_json_calls += 1
        return super().get_json(ref)


class FailingResolveArtifactStore(ArtifactStore):
    def get_json(self, ref):
        raise RuntimeError("artifact resolve failed")


class InvalidTypeResolveArtifactStore(ArtifactStore):
    def get_json(self, ref):
        return {"not": "a list"}


class MismatchedResolveArtifactStore(ArtifactStore):
    def get_json(self, ref):
        return [
            {
                "document_id": "tampered",
                "text": "tampered",
                "score": 1.0,
            }
        ]


class RecordingExecutorAgent(ExecutorAgent):
    def __init__(self) -> None:
        self.handle_calls = 0

    def handle(self, envelope: AgentEnvelope, ctx: RunContext) -> AgentEnvelope:
        self.handle_calls += 1
        return super().handle(envelope, ctx)


def _events(path: Path) -> list[TraceEvent]:
    if not path.exists():
        return []
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _business_events(path: Path) -> list[TraceEvent]:
    return [
        event
        for event in _events(path)
        if event.message_type in {
            MessageType.REQUEST.value,
            MessageType.RESULT.value,
        }
    ]


def _agents(executor: ExecutorAgent | None = None) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent() if executor is None else executor,
        SummarizerAgent(),
    ]:
        registry.register(agent)
    return registry


def _provider_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={
                "network": "Network troubleshooting completed.",
            },
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(),
    )


def _memory_service(tmp_path: Path) -> MemoryService:
    dim = 8
    return MemoryService(
        SQLiteMemoryStore(tmp_path / "memory"),
        HashEmbeddingProvider(dim=dim),
        VectorIndex(dim),
    )


def _context(
    tmp_path: Path,
    *,
    name: str,
    mode: RunMode,
    artifact_store: object,
    metrics: MetricsCollector | object | None = None,
    state_hub: object | None = None,
    memory_service: object | None = None,
) -> tuple[CapabilityRegistry, MetricsCollector | object, TraceLogger, Path, RunContext]:
    capability_registry = CapabilityRegistry()
    resolved_metrics = MetricsCollector() if metrics is None else metrics
    trace_path = tmp_path / f"{name}.jsonl"
    trace_logger = TraceLogger(trace_path)
    poison = PoisonService()
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=mode,
        config=AgentIPCConfig(),
        registry=capability_registry,
        state_hub=poison if state_hub is None else state_hub,
        artifact_store=artifact_store,  # type: ignore[arg-type]
        memory_service=poison if memory_service is None else memory_service,
        metrics=resolved_metrics,  # type: ignore[arg-type]
        trace_logger=trace_logger,
        provider_bundle=_provider_bundle(),
        use_state=False,
        use_memory=False,
        use_sandbox=False,
    )
    return capability_registry, resolved_metrics, trace_logger, trace_path, ctx


def test_structured_artifact_flow_round_trips_real_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_store = RecordingArtifactStore(tmp_path / "artifacts")
    registry = _agents()
    capabilities, metrics, trace_logger, trace_path, ctx = _context(
        tmp_path,
        name="structured-artifact",
        mode=RunMode.STRUCTURED,
        artifact_store=artifact_store,
    )
    assert isinstance(metrics, MetricsCollector)

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

    handshake = run_handshake(
        agent_registry=registry,
        capability_registry=capabilities,
        trace_logger=trace_logger,
        trace_id=ctx.trace_id,
        task_id=ctx.task_id,
    )
    assert len(handshake) == 12

    final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

    assert final.result is not None
    assert final.result["answer"] == "Network troubleshooting completed."
    assert "network-manager" in final.result["evidence_summary"]
    execution = final.result["memory_candidate"]["payload"]["execution"]
    assert execution["operation"] == "identity"
    assert "network-manager" in execution["output"]["retrieved_document_ids"]

    snapshot = metrics.snapshot()
    assert snapshot.message_count == 8
    assert snapshot.protocol_bytes > 0
    assert snapshot.artifact_ref_count == 1
    assert snapshot.state_transfer_count == 0
    assert snapshot.memory_retrieved == 0
    assert snapshot.memory_used == 0

    business = _business_events(trace_path)
    assert len(business) == 8
    assert [(event.message_type, event.action) for event in business] == [
        ("REQUEST", "PLAN"),
        ("RESULT", "PLAN"),
        ("REQUEST", "RETRIEVE"),
        ("RESULT", "RETRIEVE"),
        ("REQUEST", "EXECUTE"),
        ("RESULT", "EXECUTE"),
        ("REQUEST", "SUMMARIZE"),
        ("RESULT", "SUMMARIZE"),
    ]

    artifact_events = [event for event in business if event.artifact_refs]
    assert len(artifact_events) == 1
    artifact_event = artifact_events[0]
    assert artifact_event.message_type == MessageType.REQUEST.value
    assert artifact_event.action == ActionType.EXECUTE.value
    assert artifact_event.state_refs == []
    assert artifact_event.memory_refs == []
    assert len(artifact_event.artifact_refs) == 1

    ref = artifact_event.artifact_refs[0]
    assert re.fullmatch(r"artifact://sha256/[0-9a-f]{64}", ref.uri)
    assert ref.media_type == "application/json"
    assert ref.size_bytes > 0
    assert ref.summary == "retriever evidence"
    assert artifact_store.exists(ref) is True

    assert artifact_store.put_json_calls == 1
    runtime_get_calls = artifact_store.get_json_calls
    assert runtime_get_calls >= 1

    stored_evidence = artifact_store.get_json(ref)
    assert type(stored_evidence) is list
    assert any(
        item["document_id"] == "network-manager"
        for item in stored_evidence
    )

    raw_trace = trace_path.read_text(encoding="utf-8")
    assert "artifact://sha256/" in raw_trace
    assert "retriever evidence" in raw_trace
    assert EVIDENCE_SENTINEL not in raw_trace

    assert len(encoded) == 8
    execute_request_envelopes = [
        (envelope, payload)
        for envelope, payload in encoded
        if envelope.message_type is MessageType.REQUEST
        and envelope.action is ActionType.EXECUTE
    ]
    assert len(execute_request_envelopes) == 1
    execute_request, execute_wire = execute_request_envelopes[0]
    assert len(execute_request.artifact_refs) == 1
    execute_wire_text = execute_wire.decode("utf-8")
    assert "artifact://sha256/" in execute_wire_text
    assert "retriever evidence" in execute_wire_text
    assert "network-manager" in execute_wire_text
    assert EVIDENCE_SENTINEL not in execute_wire_text


def test_text_mode_materializes_artifact_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_store = RecordingArtifactStore(tmp_path / "text-artifacts")
    state_hub = StateHub(transport="inproc")
    memory_service = _memory_service(tmp_path / "text-services")
    registry = _agents()
    _, metrics, _, _, ctx = _context(
        tmp_path,
        name="text-artifact",
        mode=RunMode.TEXT,
        artifact_store=artifact_store,
        state_hub=state_hub,
        memory_service=memory_service,
    )
    assert isinstance(metrics, MetricsCollector)

    resolver = ReferenceResolver(
        state_hub=state_hub,
        artifact_store=artifact_store,
        memory_service=memory_service,
    )
    rendered: list[tuple[AgentEnvelope, str]] = []

    def recording_render(
        envelope: AgentEnvelope,
        resolver: ReferenceResolver | None = None,
    ) -> str:
        text = real_render(envelope, resolver=resolver)
        rendered.append((envelope, text))
        return text

    monkeypatch.setattr(
        "agentipc.runtime.text_transport.render",
        recording_render,
    )

    final = Orchestrator(
        Router(registry),
        text_transport=TextTransport(
            text_counter=TextCounter(use_tiktoken=False),
            resolver=resolver,
        ),
    ).run_task(
        task=TASK,
        ctx=ctx,
    )

    assert final.result is not None
    assert final.result["answer"] == "Network troubleshooting completed."
    assert "network-manager" in final.result["evidence_summary"]

    execute_request_texts = [
        text
        for envelope, text in rendered
        if envelope.message_type is MessageType.REQUEST
        and envelope.action is ActionType.EXECUTE
    ]
    assert len(execute_request_texts) == 1
    execute_text = execute_request_texts[0]
    assert "Artifact references:" in execute_text
    assert "artifact://sha256/" in execute_text
    assert EVIDENCE_SENTINEL in execute_text

    snapshot = metrics.snapshot()
    assert snapshot.message_count == 8
    assert snapshot.protocol_bytes == 0
    assert snapshot.text_chars > 0


def test_poison_artifact_store_keeps_inline_compatibility(tmp_path: Path) -> None:
    registry = _agents()
    poison = PoisonService()
    _, metrics, _, trace_path, ctx = _context(
        tmp_path,
        name="inline-compat",
        mode=RunMode.STRUCTURED,
        artifact_store=poison,
    )
    assert isinstance(metrics, MetricsCollector)

    final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

    assert final.result is not None
    assert final.result["answer"] == "Network troubleshooting completed."
    assert metrics.snapshot().message_count == 8
    assert metrics.snapshot().artifact_ref_count == 0
    assert all(
        event.artifact_refs == []
        for event in _business_events(trace_path)
    )


def test_artifact_resolve_failure_propagates_before_executor(
    tmp_path: Path,
) -> None:
    executor = RecordingExecutorAgent()
    registry = _agents(executor)
    artifact_store = FailingResolveArtifactStore(tmp_path / "failing-artifacts")
    _, _, _, _, ctx = _context(
        tmp_path,
        name="resolve-failure",
        mode=RunMode.STRUCTURED,
        artifact_store=artifact_store,
    )

    with pytest.raises(RuntimeError, match="artifact resolve failed"):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

    assert executor.handle_calls == 0


def test_resolved_artifact_must_be_a_list(tmp_path: Path) -> None:
    registry = _agents()
    artifact_store = InvalidTypeResolveArtifactStore(
        tmp_path / "invalid-type-artifacts"
    )
    _, _, _, _, ctx = _context(
        tmp_path,
        name="resolve-type",
        mode=RunMode.STRUCTURED,
        artifact_store=artifact_store,
    )

    with pytest.raises(
        ValueError,
        match="resolved artifact evidence must be a list",
    ):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)


def test_resolved_artifact_must_equal_retriever_evidence(tmp_path: Path) -> None:
    registry = _agents()
    artifact_store = MismatchedResolveArtifactStore(
        tmp_path / "mismatched-artifacts"
    )
    _, _, _, _, ctx = _context(
        tmp_path,
        name="resolve-mismatch",
        mode=RunMode.STRUCTURED,
        artifact_store=artifact_store,
    )

    with pytest.raises(
        ValueError,
        match="does not match retriever evidence",
    ):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
