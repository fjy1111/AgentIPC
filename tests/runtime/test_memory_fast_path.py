from __future__ import annotations

from pathlib import Path

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router
from agentipc.state.hub import StateHub


TASK = "exact repeated task"
ANSWER = "validated cached answer"


def _agents() -> AgentRegistry:
    registry = AgentRegistry()
    registry.register(PlannerAgent())
    registry.register(RetrieverAgent(knowledge=[{
        "document_id": "doc-1",
        "text": "exact repeated task evidence",
        "keywords": ["exact", "repeated"],
    }]))
    registry.register(ExecutorAgent())
    registry.register(SummarizerAgent())
    return registry


def _bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={"exact repeated task": ANSWER},
            default_text=ANSWER,
        ),
        embedding=HashEmbeddingProvider(dim=32),
    )


def _ctx(tmp_path: Path, *, name: str, service: MemoryService, bundle: ProviderBundle):
    metrics = MetricsCollector()
    hub = StateHub(transport="inproc")
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(),
        registry=CapabilityRegistry(),
        state_hub=hub,
        artifact_store=ArtifactStore(tmp_path / f"artifacts-{name}"),
        memory_service=service,
        metrics=metrics,
        trace_logger=TraceLogger(tmp_path / f"{name}.jsonl"),
        provider_bundle=bundle,
        use_state=False,
        use_memory=True,
        use_sandbox=False,
    )
    return metrics, hub, ctx


def test_exact_validated_result_skips_full_pipeline(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    embedding = HashEmbeddingProvider(dim=32)
    service = MemoryService(store, embedding, VectorIndex(embedding.dim))
    bundle = _bundle()
    try:
        first_metrics, first_hub, first_ctx = _ctx(
            tmp_path, name="first", service=service, bundle=bundle
        )
        try:
            first = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=first_ctx)
        finally:
            first_hub.close()
        assert first.result is not None
        assert first.result["answer"] == ANSWER
        assert first_metrics.snapshot().llm_call_count == 2
        service.mark_validated_result("mem_task-first", passed=True)

        second_metrics, second_hub, second_ctx = _ctx(
            tmp_path, name="second", service=service, bundle=bundle
        )
        try:
            second = Orchestrator(
                Router(_agents()), memory_fast_path=True
            ).run_task(task=TASK, ctx=second_ctx)
        finally:
            second_hub.close()
        assert second.result is not None
        assert second.result["answer"] == ANSWER
        snapshot = second_metrics.snapshot()
        assert snapshot.fast_path_hit_count == 1
        assert snapshot.llm_call_count == 0
        assert snapshot.message_count == 0
        assert snapshot.tool_call_count == 0
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 1
        assert snapshot.memory_effective == 1
        assert snapshot.memory_harmful == 0
        assert service.get("mem_task-second") is None
    finally:
        store.close()


def test_unvalidated_result_does_not_take_fast_path(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    embedding = HashEmbeddingProvider(dim=32)
    service = MemoryService(store, embedding, VectorIndex(embedding.dim))
    bundle = _bundle()
    try:
        first_metrics, first_hub, first_ctx = _ctx(
            tmp_path, name="unvalidated-first", service=service, bundle=bundle
        )
        try:
            Orchestrator(Router(_agents())).run_task(task=TASK, ctx=first_ctx)
        finally:
            first_hub.close()
        assert first_metrics.snapshot().llm_call_count == 2

        second_metrics, second_hub, second_ctx = _ctx(
            tmp_path, name="unvalidated-second", service=service, bundle=bundle
        )
        try:
            Orchestrator(
                Router(_agents()), memory_fast_path=True
            ).run_task(task=TASK, ctx=second_ctx)
        finally:
            second_hub.close()
        assert second_metrics.snapshot().fast_path_hit_count == 0
        assert second_metrics.snapshot().llm_call_count == 2
    finally:
        store.close()
