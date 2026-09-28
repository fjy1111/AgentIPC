from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router


TASK = "diagnose openEuler network connectivity"
TASK_ID = "task-memory-write"
MEMORY_ID = "mem_task-memory-write"
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


class RecordingMemoryService(MemoryService):
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedding_provider: HashEmbeddingProvider,
        vector_index: VectorIndex,
    ) -> None:
        super().__init__(store, embedding_provider, vector_index)
        self.retrieve_calls = 0
        self.write_calls = 0
        self.mark_used_calls = 0
        self.written_records: list[MemoryRecord] = []
        self.stored_records: list[MemoryRecord] = []
        self.call_order: list[str] = []

    def retrieve(self, query: str, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.retrieve_calls += 1
        self.call_order.append("retrieve")
        return super().retrieve(query, **kwargs)

    def write(self, record: MemoryRecord) -> MemoryRecord:
        self.write_calls += 1
        self.call_order.append("write")
        self.written_records.append(record.model_copy(deep=True))
        stored = super().write(record)
        self.stored_records.append(stored.model_copy(deep=True))
        return stored

    def mark_used(
        self,
        memory_id: str,
        *,
        effective: bool | None = None,
    ) -> None:
        self.mark_used_calls += 1
        return super().mark_used(memory_id, effective=effective)


class FailingWriteMemoryService(MemoryService):
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedding_provider: HashEmbeddingProvider,
        vector_index: VectorIndex,
    ) -> None:
        super().__init__(store, embedding_provider, vector_index)
        self.retrieve_calls = 0
        self.write_calls = 0
        self.mark_used_calls = 0

    def retrieve(self, query: str, **kwargs: Any):  # type: ignore[no-untyped-def]
        self.retrieve_calls += 1
        return super().retrieve(query, **kwargs)

    def write(self, record: MemoryRecord) -> MemoryRecord:
        self.write_calls += 1
        raise RuntimeError("memory write failed")

    def mark_used(
        self,
        memory_id: str,
        *,
        effective: bool | None = None,
    ) -> None:
        self.mark_used_calls += 1
        return super().mark_used(memory_id, effective=effective)


class InvalidWriteReturnMemoryService(RecordingMemoryService):
    def write(self, record: MemoryRecord):  # type: ignore[no-untyped-def]
        super().write(record)
        return object()


class InvalidCandidateSummarizer(SummarizerAgent):
    def __init__(self, candidate: object) -> None:
        self._candidate = candidate

    def handle(self, envelope: AgentEnvelope, ctx: RunContext) -> AgentEnvelope:
        response = super().handle(envelope, ctx)
        assert response.result is not None
        response.result["memory_candidate"] = self._candidate
        return response


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
        embedding=HashEmbeddingProvider(dim=EMBEDDING_DIM),
    )


def _agents(
    *,
    summarizer: SummarizerAgent | None = None,
) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent(),
        SummarizerAgent() if summarizer is None else summarizer,
    ]:
        registry.register(agent)
    return registry


def _context(
    tmp_path: Path,
    *,
    name: str,
    use_memory: bool,
    memory_service: object,
    task_id: str = TASK_ID,
) -> tuple[MetricsCollector, Path, RunContext]:
    metrics = MetricsCollector()
    trace_path = tmp_path / f"{name}.jsonl"
    poison = PoisonService()
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=task_id,
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(),
        registry=CapabilityRegistry(),
        state_hub=poison,  # type: ignore[arg-type]
        artifact_store=poison,  # type: ignore[arg-type]
        memory_service=memory_service,  # type: ignore[arg-type]
        metrics=metrics,
        trace_logger=TraceLogger(trace_path),
        provider_bundle=_provider_bundle(),
        use_state=False,
        use_memory=use_memory,
        use_sandbox=False,
    )
    return metrics, trace_path, ctx


def _recording_service(
    root: Path,
) -> tuple[SQLiteMemoryStore, RecordingMemoryService]:
    store = SQLiteMemoryStore(root)
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = RecordingMemoryService(store, provider, VectorIndex(provider.dim))
    return store, service


def test_memory_enabled_task_writes_candidate_once_and_persists_across_reopen(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "memory"
    store, service = _recording_service(memory_root)
    try:
        assert service.store.list_records() == []

        metrics, trace_path, ctx = _context(
            tmp_path,
            name="memory-write",
            use_memory=True,
            memory_service=service,
        )
        final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert final.result is not None
        assert final.result["answer"] == FINAL_ANSWER
        candidate = final.result["memory_candidate"]
        assert type(candidate) is dict
        candidate_before = deepcopy(candidate)
        assert set(final.result) == {"answer", "evidence_summary", "memory_candidate"}

        assert service.retrieve_calls == 1
        assert service.write_calls == 1
        assert service.mark_used_calls == 0
        assert service.call_order == ["retrieve", "write"]

        assert len(service.written_records) == 1
        write_input = service.written_records[0]
        assert write_input.memory_id == MEMORY_ID
        assert write_input.embedding is None
        assert write_input.reuse_count == 0
        assert write_input.success_count == 0
        assert write_input.failure_count == 0
        assert write_input.last_accessed_at is None

        stored = service.get(MEMORY_ID)
        assert isinstance(stored, MemoryRecord)
        assert stored.source_agent == "summarizer"
        assert stored.task_topic == TASK
        assert stored.summary == FINAL_ANSWER
        assert stored.memory_type is MemoryType.RESULT
        assert stored.tags == []
        assert stored.keywords == []
        assert stored.payload["answer"] == final.result["answer"]
        assert stored.payload["evidence_summary"] == final.result["evidence_summary"]
        assert stored.payload["execution"] == candidate["payload"]["execution"]
        assert stored.embedding is not None
        assert len(stored.embedding) == service.embedding_provider.dim
        assert stored.reuse_count == 0
        assert stored.success_count == 0
        assert stored.failure_count == 0
        assert stored.last_accessed_at is None

        assert stored.source_agent == candidate["source_agent"]
        assert stored.task_topic == candidate["task_topic"]
        assert stored.summary == candidate["summary"]
        assert stored.memory_type.value == candidate["memory_type"]
        assert stored.tags == candidate["tags"]
        assert stored.keywords == candidate["keywords"]
        assert stored.payload == candidate["payload"]

        assert final.result["memory_candidate"] == candidate_before
        assert "memory_id" not in candidate
        assert "embedding" not in candidate
        assert "reuse_count" not in candidate

        snapshot = metrics.snapshot()
        assert snapshot.message_count == 8
        assert snapshot.memory_retrieved == 0
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert snapshot.state_transfer_count == 0
        assert snapshot.artifact_ref_count == 0

        business = _events(trace_path)
        assert len(business) == 8
        assert business[-1].action == ActionType.SUMMARIZE.value
        assert all(event.action != ActionType.MEMORY_WRITE.value for event in business)
    finally:
        store.close()

    reopened_store = SQLiteMemoryStore(memory_root)
    reopened_provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    reopened_service = MemoryService(
        reopened_store,
        reopened_provider,
        VectorIndex(reopened_provider.dim),
    )
    try:
        persisted = reopened_service.get(MEMORY_ID)
        assert isinstance(persisted, MemoryRecord)
        assert persisted.source_agent == "summarizer"
        assert persisted.task_topic == TASK
        assert persisted.summary == FINAL_ANSWER
        assert persisted.memory_type is MemoryType.RESULT
        assert persisted.embedding is not None

        refs = reopened_service.retrieve(TASK)
        assert any(ref.memory_id == MEMORY_ID for ref in refs)
    finally:
        reopened_store.close()


def test_use_memory_false_neither_retrieves_nor_writes(tmp_path: Path) -> None:
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="memory-disabled",
        use_memory=False,
        memory_service=PoisonService(),
    )

    final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)
    assert final.result is not None
    assert final.result["answer"] == FINAL_ANSWER

    snapshot = metrics.snapshot()
    assert snapshot.message_count == 8
    assert snapshot.memory_retrieved == 0
    assert snapshot.memory_used == 0
    assert snapshot.memory_effective == 0
    assert snapshot.memory_harmful == 0
    assert len(_events(trace_path)) == 8


def test_write_failure_propagates_after_business_chain_completes(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "failing-memory")
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = FailingWriteMemoryService(store, provider, VectorIndex(provider.dim))
    try:
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="write-failure",
            use_memory=True,
            memory_service=service,
        )

        with pytest.raises(RuntimeError, match="memory write failed"):
            Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert service.retrieve_calls == 1
        assert service.write_calls == 1
        assert service.mark_used_calls == 0
        snapshot = metrics.snapshot()
        assert snapshot.message_count == 8
        assert snapshot.memory_retrieved == 0
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        business = _events(trace_path)
        assert len(business) == 8
        assert business[-1].action == ActionType.SUMMARIZE.value
    finally:
        store.close()


def test_invalid_write_return_is_rejected(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "invalid-write-return")
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = InvalidWriteReturnMemoryService(
        store,
        provider,
        VectorIndex(provider.dim),
    )
    try:
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="invalid-write-return",
            use_memory=True,
            memory_service=service,
        )

        with pytest.raises(
            ValueError,
            match=r"MemoryService\.write\(\) must return a MemoryRecord",
        ):
            Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert service.retrieve_calls == 1
        assert service.write_calls == 1
        assert service.mark_used_calls == 0
        assert service.get(MEMORY_ID) is not None
        assert metrics.snapshot().message_count == 8
        assert len(_events(trace_path)) == 8
    finally:
        store.close()


@pytest.mark.parametrize("candidate", [None, "invalid"])
def test_invalid_memory_candidate_is_rejected_without_write(
    tmp_path: Path,
    candidate: object,
) -> None:
    store, service = _recording_service(tmp_path / f"invalid-candidate-{candidate}")
    try:
        metrics, trace_path, ctx = _context(
            tmp_path,
            name=f"invalid-candidate-{candidate}",
            use_memory=True,
            memory_service=service,
        )

        with pytest.raises(
            ValueError,
            match=r"summarizer result requires dict result\['memory_candidate'\]",
        ):
            Orchestrator(
                Router(_agents(summarizer=InvalidCandidateSummarizer(candidate)))
            ).run_task(task=TASK, ctx=ctx)

        assert service.retrieve_calls == 1
        assert service.write_calls == 0
        assert service.mark_used_calls == 0
        assert service.call_order == ["retrieve"]
        assert metrics.snapshot().message_count == 8
        assert len(_events(trace_path)) == 8
        assert service.store.list_records() == []
    finally:
        store.close()
