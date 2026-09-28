from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router
from agentipc.state.hub import StateHub


TASK = "diagnostic evidence"
PLANNER_NOTE = "State-aware retrieval completed."
SHARED_TEXT = "shared diagnostic evidence"

EXPECTED_PLAN = {
    "task": TASK,
    "steps": ["retrieve", "execute", "summarize"],
    "required_capabilities": ["retrieve", "execute", "summarize"],
    "retrieval_topics": [TASK],
    "requires_execution": False,
    "planner_note": PLANNER_NOTE,
}

DIFFERENT_PLAN = {
    "task": "filesystem recovery",
    "steps": ["other"],
    "required_capabilities": ["other"],
    "retrieval_topics": ["filesystem"],
    "requires_execution": True,
    "planner_note": "different profile",
}

KNOWLEDGE = [
    {
        "document_id": "doc-a",
        "text": SHARED_TEXT,
        "keywords": ["diagnostic", "evidence"],
        "state_profile": DIFFERENT_PLAN,
    },
    {
        "document_id": "doc-z",
        "text": SHARED_TEXT,
        "keywords": ["diagnostic", "evidence"],
        "state_profile": EXPECTED_PLAN,
    },
]


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"forbidden infrastructure access: {name}")


class RecordingStateHub(StateHub):
    def __init__(self, *, transport: str = "inproc") -> None:
        super().__init__(transport=transport)
        self.put_calls = 0
        self.resolve_calls = 0
        self.release_calls = []
        self.close_calls = 0
        self.created_refs = []
        self.resolved_refs = []
        self.shm_segment_seen_during_resolve = False

    def put_array(
        self,
        array: np.ndarray,
        *,
        kind: str,
        summary: str,
    ):
        self.put_calls += 1
        ref = super().put_array(array, kind=kind, summary=summary)
        self.created_refs.append(ref)
        return ref

    def resolve_array(self, ref):
        self.resolve_calls += 1
        self.resolved_refs.append(ref)
        if ref.transport == "shm":
            name = ref.uri.removeprefix("shm://agentipc/")
            segment = SharedMemory(name=name)
            segment.close()
            self.shm_segment_seen_during_resolve = True
        return super().resolve_array(ref)

    def release(self, ref) -> None:
        self.release_calls.append(ref)
        super().release(ref)

    def close(self) -> None:
        self.close_calls += 1
        super().close()


class FailingResolveStateHub(RecordingStateHub):
    def resolve_array(self, ref):
        self.resolve_calls += 1
        self.resolved_refs.append(ref)
        raise RuntimeError("state resolve failed")


class FailingPutStateHub(RecordingStateHub):
    def put_array(
        self,
        array: np.ndarray,
        *,
        kind: str,
        summary: str,
    ):
        self.put_calls += 1
        raise RuntimeError("state put failed")


class FailingReleaseStateHub(RecordingStateHub):
    def release(self, ref) -> None:
        self.release_calls.append(ref)
        raise RuntimeError("state release failed")


class RecordingArtifactStore(ArtifactStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.put_refs = []

    def put_json(self, value, *, summary: str):
        ref = super().put_json(value, summary=summary)
        self.put_refs.append(ref)
        return ref


class FailingRetrieveMemoryService(MemoryService):
    def retrieve(self, query: str, **kwargs):
        raise RuntimeError("memory retrieve failed")


def _agents() -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent(),
        SummarizerAgent(),
    ]:
        registry.register(agent)
    return registry


def _provider_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={TASK: PLANNER_NOTE},
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(),
    )


def _context(
    tmp_path: Path,
    *,
    name: str,
    state_hub: object,
    use_state: bool = True,
    artifact_store: object | None = None,
    memory_service: object | None = None,
    use_memory: bool = False,
) -> tuple[MetricsCollector, Path, RunContext]:
    metrics = MetricsCollector()
    trace_path = tmp_path / f"{name}.jsonl"
    poison = PoisonService()
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(),
        registry=CapabilityRegistry(),
        state_hub=state_hub,  # type: ignore[arg-type]
        artifact_store=(
            poison if artifact_store is None else artifact_store
        ),  # type: ignore[arg-type]
        memory_service=(
            poison if memory_service is None else memory_service
        ),  # type: ignore[arg-type]
        metrics=metrics,
        trace_logger=TraceLogger(trace_path),
        provider_bundle=_provider_bundle(),
        use_state=use_state,
        use_memory=use_memory,
        use_sandbox=False,
    )
    return metrics, trace_path, ctx


def _events(path: Path) -> list[TraceEvent]:
    if not path.exists():
        return []
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _assert_final_answer(final: AgentEnvelope) -> None:
    assert final.result is not None
    assert final.result["answer"] == PLANNER_NOTE


def _new_memory_service(
    root: Path,
    *,
    service_type: type[MemoryService] = MemoryService,
) -> tuple[SQLiteMemoryStore, MemoryService]:
    store = SQLiteMemoryStore(root)
    provider = HashEmbeddingProvider()
    service = service_type(store, provider, VectorIndex(provider.dim))
    return store, service


def test_successful_inproc_task_releases_only_task_state_and_keeps_trace_metadata(
    tmp_path: Path,
) -> None:
    state_hub = RecordingStateHub(transport="inproc")
    try:
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="inproc-success",
            state_hub=state_hub,
        )
        final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)
        _assert_final_answer(final)

        assert state_hub.put_calls == 1
        assert state_hub.resolve_calls == 1
        assert len(state_hub.release_calls) == 1
        assert len(state_hub.created_refs) == 1
        ref = state_hub.created_refs[0]
        assert state_hub.release_calls == [ref]
        assert ref.transport == "inproc"
        assert state_hub.exists(ref) is False
        assert state_hub.close_calls == 0

        snapshot = metrics.snapshot()
        assert snapshot.message_count == 8
        assert snapshot.state_transfer_count == 1
        assert snapshot.state_bytes == 256

        events = _events(trace_path)
        assert len(events) == 8
        state_events = [event for event in events if event.state_refs]
        assert len(state_events) == 1
        assert state_events[0].message_type == MessageType.REQUEST.value
        assert state_events[0].action == ActionType.RETRIEVE.value
        assert state_events[0].state_refs == [ref]
    finally:
        state_hub.close()


def test_successful_shm_task_unlinks_real_os_segment(tmp_path: Path) -> None:
    state_hub = RecordingStateHub(transport="shm")
    try:
        metrics, _, ctx = _context(
            tmp_path,
            name="shm-success",
            state_hub=state_hub,
        )
        final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)
        _assert_final_answer(final)

        assert state_hub.put_calls == 1
        assert state_hub.resolve_calls == 1
        assert len(state_hub.release_calls) == 1
        assert state_hub.shm_segment_seen_during_resolve is True
        ref = state_hub.created_refs[0]
        assert ref.transport == "shm"
        assert state_hub.exists(ref) is False
        assert state_hub.close_calls == 0
        assert metrics.snapshot().state_bytes == 256

        name = ref.uri.removeprefix("shm://agentipc/")
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name)
    finally:
        state_hub.close()


def test_state_resolve_failure_releases_state_and_preserves_original_error(
    tmp_path: Path,
) -> None:
    state_hub = FailingResolveStateHub()
    try:
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="resolve-failure",
            state_hub=state_hub,
        )
        with pytest.raises(RuntimeError, match="state resolve failed"):
            Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert state_hub.put_calls == 1
        assert state_hub.resolve_calls == 1
        assert len(state_hub.release_calls) == 1
        ref = state_hub.created_refs[0]
        assert state_hub.release_calls == [ref]
        assert state_hub.exists(ref) is False
        assert state_hub.close_calls == 0

        snapshot = metrics.snapshot()
        assert snapshot.message_count == 3
        assert snapshot.state_transfer_count == 1
        assert snapshot.state_bytes == 256
        assert len(_events(trace_path)) == 3
    finally:
        state_hub.close()


def test_preexisting_owned_state_survives_task_cleanup(tmp_path: Path) -> None:
    state_hub = RecordingStateHub()
    try:
        preexisting_ref = state_hub.put_array(
            np.array([7.0], dtype=np.float32),
            kind="external-to-task",
            summary="caller-owned state",
        )
        state_hub.release_calls.clear()

        _, _, ctx = _context(
            tmp_path,
            name="ownership",
            state_hub=state_hub,
        )
        final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)
        _assert_final_answer(final)

        task_ref = state_hub.created_refs[-1]
        assert task_ref != preexisting_ref
        assert state_hub.release_calls == [task_ref]
        assert state_hub.exists(task_ref) is False
        assert state_hub.exists(preexisting_ref) is True
        assert state_hub.close_calls == 0
    finally:
        state_hub.close()


def test_same_state_hub_is_reusable_across_two_tasks(tmp_path: Path) -> None:
    state_hub = RecordingStateHub()
    try:
        for name in ["reuse-a", "reuse-b"]:
            _, _, ctx = _context(
                tmp_path,
                name=name,
                state_hub=state_hub,
            )
            final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)
            _assert_final_answer(final)

        assert state_hub.put_calls == 2
        assert state_hub.resolve_calls == 2
        assert len(state_hub.release_calls) == 2
        assert state_hub.close_calls == 0
        assert len(state_hub.created_refs) == 2
        assert all(state_hub.exists(ref) is False for ref in state_hub.created_refs)
    finally:
        state_hub.close()


def test_use_state_false_does_not_access_or_cleanup_state_service(
    tmp_path: Path,
) -> None:
    metrics, _, ctx = _context(
        tmp_path,
        name="state-disabled",
        state_hub=PoisonService(),
        use_state=False,
    )
    final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)
    _assert_final_answer(final)
    snapshot = metrics.snapshot()
    assert snapshot.message_count == 8
    assert snapshot.state_transfer_count == 0
    assert snapshot.state_bytes == 0


def test_failure_before_state_creation_does_not_release(tmp_path: Path) -> None:
    state_hub = RecordingStateHub()
    store, service = _new_memory_service(
        tmp_path / "pre-state-memory",
        service_type=FailingRetrieveMemoryService,
    )
    try:
        metrics, trace_path, ctx = _context(
            tmp_path,
            name="pre-state-failure",
            state_hub=state_hub,
            memory_service=service,
            use_memory=True,
        )
        with pytest.raises(RuntimeError, match="memory retrieve failed"):
            Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert state_hub.put_calls == 0
        assert state_hub.resolve_calls == 0
        assert state_hub.release_calls == []
        assert state_hub.close_calls == 0
        assert metrics.snapshot().message_count == 2
        assert len(_events(trace_path)) == 2
    finally:
        store.close()
        state_hub.close()


def test_put_array_failure_has_no_fake_release(tmp_path: Path) -> None:
    state_hub = FailingPutStateHub()
    try:
        _, _, ctx = _context(
            tmp_path,
            name="put-failure",
            state_hub=state_hub,
        )
        with pytest.raises(RuntimeError, match="state put failed"):
            Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert state_hub.put_calls == 1
        assert state_hub.created_refs == []
        assert state_hub.release_calls == []
        assert state_hub.close_calls == 0
    finally:
        state_hub.close()


def test_release_failure_propagates_and_does_not_close_hub(tmp_path: Path) -> None:
    state_hub = FailingReleaseStateHub()
    try:
        metrics, _, ctx = _context(
            tmp_path,
            name="release-failure",
            state_hub=state_hub,
        )
        with pytest.raises(RuntimeError, match="state release failed"):
            Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert state_hub.put_calls == 1
        assert state_hub.resolve_calls == 1
        assert len(state_hub.release_calls) == 1
        ref = state_hub.created_refs[0]
        assert state_hub.release_calls == [ref]
        assert state_hub.exists(ref) is True
        assert state_hub.close_calls == 0
        assert metrics.snapshot().message_count == 8
    finally:
        state_hub.close()


def test_artifact_persists_after_state_cleanup(tmp_path: Path) -> None:
    state_hub = RecordingStateHub()
    artifact_store = RecordingArtifactStore(tmp_path / "artifacts")
    try:
        metrics, _, ctx = _context(
            tmp_path,
            name="artifact-persistence",
            state_hub=state_hub,
            artifact_store=artifact_store,
        )
        final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)
        _assert_final_answer(final)

        task_ref = state_hub.created_refs[0]
        assert state_hub.exists(task_ref) is False
        assert len(artifact_store.put_refs) == 1
        assert artifact_store.exists(artifact_store.put_refs[0]) is True
        assert metrics.snapshot().artifact_ref_count == 1
        assert state_hub.close_calls == 0
    finally:
        state_hub.close()


def test_memory_persists_and_store_remains_readable_after_state_cleanup(
    tmp_path: Path,
) -> None:
    state_hub = RecordingStateHub()
    store, service = _new_memory_service(tmp_path / "memory")
    try:
        metrics, _, ctx = _context(
            tmp_path,
            name="memory-persistence",
            state_hub=state_hub,
            memory_service=service,
            use_memory=True,
        )
        final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)
        _assert_final_answer(final)

        task_ref = state_hub.created_refs[0]
        assert state_hub.exists(task_ref) is False
        current = service.get("mem_task-memory-persistence")
        assert current is not None
        assert current.task_topic == TASK
        records = store.list_records()
        assert len(records) == 1
        assert records[0].memory_id == current.memory_id
        assert service.get(current.memory_id) == current
        assert metrics.snapshot().memory_retrieved == 0
        assert state_hub.close_calls == 0
    finally:
        store.close()
        state_hub.close()
