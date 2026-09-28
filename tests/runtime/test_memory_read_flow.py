from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.text_counter import TextCounter
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.codec import encode as real_encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageType
from agentipc.protocol.refs import MemoryRef
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router
from agentipc.runtime.text_transport import TextTransport
from agentipc.state.hub import StateHub


TASK = "diagnose openEuler network connectivity"
MEMORY_ID = "mem-task-1-network"
MEMORY_PAYLOAD_SENTINEL = "TASK-1-MEMORY-PAYLOAD-SENTINEL"
FINAL_ANSWER = "Network troubleshooting completed."
EMBEDDING_DIM = 64

KNOWLEDGE = [
    {
        "document_id": "network-manager",
        "text": "NetworkManager manages openEuler network connections and connectivity.",
        "keywords": ["NetworkManager", "openEuler", "network connectivity"],
    },
    {
        "document_id": "filesystem",
        "text": "Use fsck for filesystem diagnostics.",
        "keywords": ["filesystem", "fsck"],
    },
]


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"forbidden infrastructure access: {name}")


class RecordingRetrieverAgent(RetrieverAgent):
    def __init__(self) -> None:
        super().__init__(KNOWLEDGE)
        self.received_memory_refs: list[list[MemoryRef]] = []

    def handle(self, envelope: AgentEnvelope, ctx: RunContext) -> AgentEnvelope:
        self.received_memory_refs.append(list(envelope.memory_refs))
        return super().handle(envelope, ctx)


class RecordingMemoryService(MemoryService):
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedding_provider: HashEmbeddingProvider,
        vector_index: VectorIndex,
    ) -> None:
        super().__init__(store, embedding_provider, vector_index)
        self.retrieve_calls = 0
        self.retrieve_queries: list[str] = []
        self.retrieve_kwargs: list[dict[str, object]] = []
        self.retrieve_results: list[list[MemoryRef]] = []
        self.mark_used_calls = 0
        self.write_calls = 0

    def retrieve(self, query: str, **kwargs: Any) -> list[MemoryRef]:
        self.retrieve_calls += 1
        self.retrieve_queries.append(query)
        self.retrieve_kwargs.append(dict(kwargs))
        refs = super().retrieve(query, **kwargs)
        self.retrieve_results.append(list(refs))
        return refs

    def write(self, record: MemoryRecord) -> MemoryRecord:
        self.write_calls += 1
        return super().write(record)

    def mark_used(
        self,
        memory_id: str,
        *,
        effective: bool | None = None,
    ) -> None:
        self.mark_used_calls += 1
        return super().mark_used(memory_id, effective=effective)


class FailingRetrieveMemoryService(MemoryService):
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedding_provider: HashEmbeddingProvider,
        vector_index: VectorIndex,
    ) -> None:
        super().__init__(store, embedding_provider, vector_index)
        self.retrieve_calls = 0

    def retrieve(self, query: str, **kwargs: Any) -> list[MemoryRef]:
        self.retrieve_calls += 1
        raise RuntimeError("memory retrieve failed")


class InvalidRetrieveMemoryService(MemoryService):
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedding_provider: HashEmbeddingProvider,
        vector_index: VectorIndex,
        returned: object,
    ) -> None:
        super().__init__(store, embedding_provider, vector_index)
        self.returned = returned
        self.retrieve_calls = 0

    def retrieve(self, query: str, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.retrieve_calls += 1
        return self.returned


def _events(path: Path) -> list[TraceEvent]:
    if not path.exists():
        return []
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _provider_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={"network": FINAL_ANSWER},
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(),
    )


def _agents() -> tuple[AgentRegistry, RecordingRetrieverAgent]:
    registry = AgentRegistry()
    retriever = RecordingRetrieverAgent()
    for agent in [
        PlannerAgent(),
        retriever,
        ExecutorAgent(),
        SummarizerAgent(),
    ]:
        registry.register(agent)
    return registry, retriever


def _context(
    tmp_path: Path,
    *,
    name: str,
    use_memory: object,
    memory_service: object,
    mode: RunMode = RunMode.STRUCTURED,
    use_state: bool = False,
    state_hub: object | None = None,
) -> tuple[MetricsCollector, Path, RunContext]:
    metrics = MetricsCollector()
    trace_path = tmp_path / f"{name}.jsonl"
    poison = PoisonService()
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=mode,
        config=AgentIPCConfig(),
        registry=CapabilityRegistry(),
        state_hub=poison if state_hub is None else state_hub,  # type: ignore[arg-type]
        artifact_store=poison,  # type: ignore[arg-type]
        memory_service=memory_service,  # type: ignore[arg-type]
        metrics=metrics,
        trace_logger=TraceLogger(trace_path),
        provider_bundle=_provider_bundle(),
        use_state=use_state,
        use_memory=use_memory,  # type: ignore[arg-type]
        use_sandbox=False,
    )
    return metrics, trace_path, ctx


def _seed_historical_memory(root: Path) -> MemoryRecord:
    store = SQLiteMemoryStore(root)
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = MemoryService(store, provider, VectorIndex(provider.dim))
    try:
        return service.write(
            MemoryRecord(
                memory_id=MEMORY_ID,
                source_agent="summarizer",
                task_topic=TASK,
                summary=TASK,
                memory_type=MemoryType.RESULT,
                tags=["openeuler", "network"],
                keywords=["networkmanager", "connectivity"],
                payload={"answer": MEMORY_PAYLOAD_SENTINEL},
            )
        )
    finally:
        store.close()


def _recording_service(root: Path) -> tuple[SQLiteMemoryStore, RecordingMemoryService]:
    store = SQLiteMemoryStore(root)
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = RecordingMemoryService(store, provider, VectorIndex(provider.dim))
    return store, service


def _plain_service(root: Path) -> tuple[SQLiteMemoryStore, MemoryService]:
    store = SQLiteMemoryStore(root)
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = MemoryService(store, provider, VectorIndex(provider.dim))
    return store, service


def test_persisted_historical_memory_is_retrieved_once_and_only_sent_to_retriever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    memory_root = tmp_path / "memory"
    seeded = _seed_historical_memory(memory_root)
    assert seeded.memory_id == MEMORY_ID

    baseline_registry, _ = _agents()
    baseline_metrics, _, baseline_ctx = _context(
        tmp_path,
        name="baseline",
        use_memory=False,
        memory_service=PoisonService(),
    )
    baseline_final = Orchestrator(Router(baseline_registry)).run_task(
        task=TASK,
        ctx=baseline_ctx,
    )
    assert baseline_metrics.snapshot().message_count == 8

    store, service = _recording_service(memory_root)
    try:
        before = service.get(MEMORY_ID)
        assert before is not None
        assert before.reuse_count == 0
        assert before.success_count == 0
        assert before.failure_count == 0
        assert before.last_accessed_at is None

        registry, retriever = _agents()
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="memory",
            use_memory=True,
            memory_service=service,
        )

        encoded: list[tuple[AgentEnvelope, bytes]] = []

        def recording_encode(envelope: AgentEnvelope) -> bytes:
            payload = real_encode(envelope)
            encoded.append((envelope.model_copy(deep=True), payload))
            return payload

        monkeypatch.setattr(
            "agentipc.runtime.orchestrator.encode",
            recording_encode,
        )

        final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

        assert final.result == baseline_final.result
        assert final.result is not None
        assert final.result["answer"] == FINAL_ANSWER

        assert service.retrieve_calls == 1
        assert service.retrieve_queries == [TASK]
        assert service.retrieve_kwargs == [{}]
        assert len(service.retrieve_results) == 1
        assert len(service.retrieve_results[0]) == 1
        hit = service.retrieve_results[0][0]
        assert isinstance(hit, MemoryRef)
        assert hit.memory_id == MEMORY_ID
        assert hit.match_type == "hybrid"
        assert hit.summary == TASK

        assert service.mark_used_calls == 0
        assert service.write_calls == 1

        assert len(retriever.received_memory_refs) == 1
        assert retriever.received_memory_refs[0] == [hit]

        snapshot = metrics.snapshot()
        assert snapshot.message_count == 8
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert snapshot.state_transfer_count == 0
        assert snapshot.state_bytes == 0
        assert snapshot.artifact_ref_count == 0

        business = _events(trace_path)
        assert len(business) == 8
        memory_events = [event for event in business if event.memory_refs]
        assert len(memory_events) == 1
        memory_event = memory_events[0]
        assert memory_event.message_type == MessageType.REQUEST.value
        assert memory_event.action == ActionType.RETRIEVE.value
        assert memory_event.memory_refs == [hit]
        assert sum(len(event.memory_refs) for event in business) == 1

        raw_trace = trace_path.read_text(encoding="utf-8")
        assert MEMORY_ID in raw_trace
        assert "hybrid" in raw_trace
        assert TASK in raw_trace
        assert MEMORY_PAYLOAD_SENTINEL not in raw_trace

        assert len(encoded) == 8
        memory_wires = [
            payload
            for envelope, payload in encoded
            if envelope.memory_refs
        ]
        assert len(memory_wires) == 1
        wire = memory_wires[0]
        assert MEMORY_ID.encode() in wire
        assert b"hybrid" in wire
        assert TASK.encode() in wire
        assert MEMORY_PAYLOAD_SENTINEL.encode() not in wire

        retrieve_requests = [
            envelope
            for envelope, _ in encoded
            if envelope.message_type is MessageType.REQUEST
            and envelope.action is ActionType.RETRIEVE
        ]
        assert len(retrieve_requests) == 1
        assert retrieve_requests[0].memory_refs == [hit]

        after = service.get(MEMORY_ID)
        assert after is not None
        assert after.reuse_count == before.reuse_count == 0
        assert after.success_count == before.success_count == 0
        assert after.failure_count == before.failure_count == 0
        assert after.last_accessed_at == before.last_accessed_at is None
    finally:
        store.close()


def test_use_memory_false_never_accesses_memory_service(tmp_path: Path) -> None:
    registry, retriever = _agents()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="memory-disabled",
        use_memory=False,
        memory_service=PoisonService(),
    )

    final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
    assert final.result is not None
    assert final.result["answer"] == FINAL_ANSWER
    assert retriever.received_memory_refs == [[]]

    snapshot = metrics.snapshot()
    assert snapshot.message_count == 8
    assert snapshot.memory_retrieved == 0
    assert snapshot.memory_used == 0
    assert snapshot.memory_effective == 0
    assert snapshot.memory_harmful == 0
    assert all(event.memory_refs == [] for event in _events(trace_path))


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_use_memory_requires_exact_bool_before_dispatch(
    tmp_path: Path,
    value: object,
) -> None:
    registry, _ = _agents()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name=f"invalid-memory-bool-{type(value).__name__}-{value}",
        use_memory=value,
        memory_service=PoisonService(),
    )

    with pytest.raises(TypeError, match="ctx.use_memory must be a bool"):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

    assert metrics.snapshot().message_count == 0
    assert _events(trace_path) == []


def test_memory_enabled_requires_real_memory_service_before_dispatch(
    tmp_path: Path,
) -> None:
    registry, _ = _agents()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="invalid-memory-service",
        use_memory=True,
        memory_service=PoisonService(),
    )

    with pytest.raises(
        TypeError,
        match="use_memory=True requires ctx.memory_service to be a MemoryService",
    ):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

    assert metrics.snapshot().message_count == 0
    assert _events(trace_path) == []


def test_text_mode_with_memory_is_rejected_before_retrieve_or_dispatch(
    tmp_path: Path,
) -> None:
    store, service = _recording_service(tmp_path / "text-memory")
    try:
        registry, _ = _agents()
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="text-memory",
            use_memory=True,
            memory_service=service,
            mode=RunMode.TEXT,
        )
        orchestrator = Orchestrator(
            Router(registry),
            text_transport=TextTransport(
                text_counter=TextCounter(use_tiktoken=False),
            ),
        )

        with pytest.raises(ValueError, match="does not support use_memory=True"):
            orchestrator.run_task(task=TASK, ctx=ctx)

        assert service.retrieve_calls == 0
        assert service.mark_used_calls == 0
        assert service.write_calls == 0
        assert metrics.snapshot().message_count == 0
        assert _events(trace_path) == []
    finally:
        store.close()


def test_empty_retrieve_continues_with_zero_memory_refs(tmp_path: Path) -> None:
    store, service = _recording_service(tmp_path / "empty-memory")
    try:
        registry, retriever = _agents()
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="empty-memory",
            use_memory=True,
            memory_service=service,
        )

        final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
        assert final.result is not None
        assert final.result["answer"] == FINAL_ANSWER
        assert service.retrieve_calls == 1
        assert service.retrieve_queries == [TASK]
        assert service.retrieve_kwargs == [{}]
        assert service.retrieve_results == [[]]
        assert retriever.received_memory_refs == [[]]
        assert metrics.snapshot().message_count == 8
        assert metrics.snapshot().memory_retrieved == 0
        assert all(event.memory_refs == [] for event in _events(trace_path))
    finally:
        store.close()


def test_retrieve_failure_propagates_without_baseline_fallback(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "failing-memory")
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = FailingRetrieveMemoryService(
        store,
        provider,
        VectorIndex(provider.dim),
    )
    try:
        registry, _ = _agents()
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="retrieve-failure",
            use_memory=True,
            memory_service=service,
        )

        with pytest.raises(RuntimeError, match="memory retrieve failed"):
            Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

        assert service.retrieve_calls == 1
        snapshot = metrics.snapshot()
        assert snapshot.message_count == 2
        assert snapshot.memory_retrieved == 0
        assert [(event.message_type, event.action) for event in _events(trace_path)] == [
            (MessageType.REQUEST.value, ActionType.PLAN.value),
            (MessageType.RESULT.value, ActionType.PLAN.value),
        ]
    finally:
        store.close()


@pytest.mark.parametrize(
    "returned",
    [
        pytest.param({}, id="dict"),
        pytest.param(None, id="none"),
        pytest.param([object()], id="non-memory-ref-item"),
    ],
)
def test_invalid_retrieve_return_is_rejected_without_silent_filtering(
    tmp_path: Path,
    returned: object,
) -> None:
    store = SQLiteMemoryStore(tmp_path / f"invalid-return-{id(returned)}")
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = InvalidRetrieveMemoryService(
        store,
        provider,
        VectorIndex(provider.dim),
        returned,
    )
    try:
        registry, _ = _agents()
        metrics, trace_path, ctx = _context(
            tmp_path,
            name=f"invalid-return-{id(returned)}",
            use_memory=True,
            memory_service=service,
        )

        with pytest.raises(
            ValueError,
            match=r"MemoryService\.retrieve\(\) must return a list\[MemoryRef\]",
        ):
            Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

        assert service.retrieve_calls == 1
        assert metrics.snapshot().message_count == 2
        assert metrics.snapshot().memory_retrieved == 0
        assert [(event.message_type, event.action) for event in _events(trace_path)] == [
            (MessageType.REQUEST.value, ActionType.PLAN.value),
            (MessageType.RESULT.value, ActionType.PLAN.value),
        ]
    finally:
        store.close()


def test_state_and_memory_paths_compose(tmp_path: Path) -> None:
    memory_root = tmp_path / "composed-memory"
    _seed_historical_memory(memory_root)
    store, service = _recording_service(memory_root)
    state_hub = StateHub(transport="inproc")
    try:
        registry, retriever = _agents()
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="state-memory",
            use_memory=True,
            memory_service=service,
            use_state=True,
            state_hub=state_hub,
        )

        final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
        assert final.result is not None
        assert final.result["answer"] == FINAL_ANSWER

        snapshot = metrics.snapshot()
        assert snapshot.message_count == 8
        assert snapshot.state_transfer_count == 1
        assert snapshot.state_bytes == 256
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert snapshot.artifact_ref_count == 0

        assert service.retrieve_calls == 1
        assert len(retriever.received_memory_refs) == 1
        assert [ref.memory_id for ref in retriever.received_memory_refs[0]] == [MEMORY_ID]

        retrieve_events = [
            event
            for event in _events(trace_path)
            if event.message_type == MessageType.REQUEST.value
            and event.action == ActionType.RETRIEVE.value
        ]
        assert len(retrieve_events) == 1
        assert len(retrieve_events[0].state_refs) == 1
        assert len(retrieve_events[0].memory_refs) == 1
        assert retrieve_events[0].memory_refs[0].memory_id == MEMORY_ID
    finally:
        state_hub.close()
        store.close()
