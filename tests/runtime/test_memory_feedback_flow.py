from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import agentipc.runtime.orchestrator as orchestrator_module
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
from agentipc.protocol.codec import encode as protocol_encode
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


TASK = "diagnose openEuler network connectivity"
EMBEDDING_DIM = 64
BASELINE_ANSWER = "BASELINE-ANSWER"
CACHED_ANSWER = "CACHED-CORRECT-ANSWER"
CACHED_SENTINEL = "MEMORY-CACHE-SENTINEL"
HISTORICAL_ANSWER = "HISTORICAL-ANSWER"
HARMFUL_ANSWER = "CURRENT-DIFFERENT-ANSWER"
HARMFUL_SENTINEL = "HARMFUL-CACHE-SENTINEL"

KNOWLEDGE = [
    {
        "document_id": "network-manager",
        "text": "NetworkManager manages openEuler network connections.",
        "keywords": ["NetworkManager", "openEuler", "network connectivity"],
    },
    {
        "document_id": "ip-route",
        "text": "The ip route command inspects the routing table.",
        "keywords": ["ip route", "routing", "connectivity"],
    },
]


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"forbidden infrastructure access: {name}")


class RecordingExecutorAgent(ExecutorAgent):
    def __init__(self) -> None:
        self.requests: list[AgentEnvelope] = []

    def handle(self, envelope: AgentEnvelope, ctx: RunContext) -> AgentEnvelope:
        self.requests.append(envelope.model_copy(deep=True))
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
        self.get_calls: list[str] = []
        self.write_calls = 0
        self.mark_used_calls: list[tuple[str, bool | None]] = []
        self.call_order: list[str] = []

    def retrieve(self, query: str, **kwargs: Any) -> list[MemoryRef]:
        self.retrieve_calls += 1
        self.call_order.append("retrieve")
        return super().retrieve(query, **kwargs)

    def get(self, memory_id: str) -> MemoryRecord | None:
        self.get_calls.append(memory_id)
        self.call_order.append(f"get:{memory_id}")
        return super().get(memory_id)

    def write(self, record: MemoryRecord) -> MemoryRecord:
        self.write_calls += 1
        self.call_order.append("write")
        return super().write(record)

    def mark_used(
        self,
        memory_id: str,
        *,
        effective: bool | None = None,
    ) -> None:
        self.mark_used_calls.append((memory_id, effective))
        self.call_order.append("mark_used")
        return super().mark_used(memory_id, effective=effective)


class FailingWriteMemoryService(RecordingMemoryService):
    def write(self, record: MemoryRecord) -> MemoryRecord:
        self.write_calls += 1
        self.call_order.append("write")
        raise RuntimeError("memory write failed")


class FailingMarkUsedMemoryService(RecordingMemoryService):
    def mark_used(
        self,
        memory_id: str,
        *,
        effective: bool | None = None,
    ) -> None:
        self.mark_used_calls.append((memory_id, effective))
        self.call_order.append("mark_used")
        raise RuntimeError("memory mark_used failed")


class MissingRecordMemoryService(RecordingMemoryService):
    def retrieve(self, query: str, **kwargs: Any) -> list[MemoryRef]:
        self.retrieve_calls += 1
        self.call_order.append("retrieve")
        return [
            MemoryRef(
                memory_id="mem-missing",
                score=1.0,
                match_type="hybrid",
                summary=query,
            )
        ]


def _record(
    memory_id: str,
    *,
    payload: dict[str, object],
    memory_type: MemoryType = MemoryType.RESULT,
    task_topic: str = TASK,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        source_agent="summarizer",
        task_topic=task_topic,
        summary=TASK,
        memory_type=memory_type,
        payload=payload,
    )


def _usable_record(
    memory_id: str = "mem-b-usable",
    *,
    answer: str = CACHED_ANSWER,
    sentinel: str = CACHED_SENTINEL,
) -> MemoryRecord:
    return _record(
        memory_id,
        payload={
            "answer": answer,
            "execution": {
                "operation": "identity",
                "output": {"cache_token": sentinel},
            },
        },
    )


def _seed(root: Path, records: list[MemoryRecord]) -> None:
    store = SQLiteMemoryStore(root)
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = MemoryService(store, provider, VectorIndex(provider.dim))
    try:
        for record in records:
            service.write(record)
    finally:
        store.close()


def _recording_service(
    root: Path,
    *,
    service_type: type[RecordingMemoryService] = RecordingMemoryService,
) -> tuple[SQLiteMemoryStore, RecordingMemoryService]:
    store = SQLiteMemoryStore(root)
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = service_type(store, provider, VectorIndex(provider.dim))
    return store, service


def _provider_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={
                CACHED_SENTINEL: CACHED_ANSWER,
                HARMFUL_SENTINEL: HARMFUL_ANSWER,
            },
            default_text=BASELINE_ANSWER,
        ),
        embedding=HashEmbeddingProvider(dim=EMBEDDING_DIM),
    )


def _agents(executor: RecordingExecutorAgent) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        executor,
        SummarizerAgent(),
    ]:
        registry.register(agent)
    return registry


def _context(
    tmp_path: Path,
    *,
    name: str,
    use_memory: bool,
    memory_service: object,
) -> tuple[MetricsCollector, Path, RunContext]:
    metrics = MetricsCollector()
    trace_path = tmp_path / f"{name}.jsonl"
    poison = PoisonService()
    return metrics, trace_path, RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
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


def _events(path: Path) -> list[TraceEvent]:
    if not path.exists():
        return []
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _execute_operation(executor: RecordingExecutorAgent) -> dict[str, object]:
    assert len(executor.requests) == 1
    operation = executor.requests[0].args["operation"]
    assert type(operation) is dict
    return operation


def _assert_four_agent_chain(trace_path: Path) -> None:
    events = _events(trace_path)
    assert len(events) == 8
    assert [(event.message_type, event.action) for event in events] == [
        (MessageType.REQUEST.value, ActionType.PLAN.value),
        (MessageType.RESULT.value, ActionType.PLAN.value),
        (MessageType.REQUEST.value, ActionType.RETRIEVE.value),
        (MessageType.RESULT.value, ActionType.RETRIEVE.value),
        (MessageType.REQUEST.value, ActionType.EXECUTE.value),
        (MessageType.RESULT.value, ActionType.EXECUTE.value),
        (MessageType.REQUEST.value, ActionType.SUMMARIZE.value),
        (MessageType.RESULT.value, ActionType.SUMMARIZE.value),
    ]
    assert all(
        event.action not in (
            ActionType.MEMORY_QUERY.value,
            ActionType.MEMORY_WRITE.value,
        )
        for event in events
    )


def test_effective_memory_is_actual_executor_input_and_feedback_is_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_executor = RecordingExecutorAgent()
    baseline_metrics, _, baseline_ctx = _context(
        tmp_path,
        name="baseline",
        use_memory=False,
        memory_service=PoisonService(),
    )
    baseline = Orchestrator(Router(_agents(baseline_executor))).run_task(
        task=TASK,
        ctx=baseline_ctx,
    )
    assert baseline.result is not None
    assert baseline.result["answer"] == BASELINE_ANSWER
    baseline_operation = _execute_operation(baseline_executor)
    assert baseline_operation["name"] == "identity"
    assert baseline_operation["value"] == {
        "retrieved_document_ids": ["network-manager", "ip-route"]
    }
    assert baseline_metrics.snapshot().memory_used == 0

    memory_root = tmp_path / "effective-memory"
    _seed(memory_root, [_usable_record()])
    store, service = _recording_service(memory_root)
    executor = RecordingExecutorAgent()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="effective",
        use_memory=True,
        memory_service=service,
    )
    encoded_execute_requests: list[bytes] = []

    def recording_encode(envelope: AgentEnvelope) -> bytes:
        payload = protocol_encode(envelope)
        if (
            envelope.message_type is MessageType.REQUEST
            and envelope.action is ActionType.EXECUTE
        ):
            encoded_execute_requests.append(payload)
        return payload

    monkeypatch.setattr(orchestrator_module, "encode", recording_encode)
    try:
        final = Orchestrator(Router(_agents(executor))).run_task(task=TASK, ctx=ctx)
        assert final.result is not None
        assert final.result["answer"] == CACHED_ANSWER

        operation = _execute_operation(executor)
        assert operation == {
            "name": "identity",
            "value": {"cache_token": CACHED_SENTINEL},
        }

        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 1
        assert snapshot.memory_effective == 1
        assert snapshot.memory_harmful == 0
        assert snapshot.message_count == 8
        assert snapshot.tool_call_count == 0
        assert snapshot.repeated_tool_call_count == 0

        assert service.retrieve_calls == 1
        assert service.get_calls == ["mem-b-usable"]
        assert service.write_calls == 1
        assert service.mark_used_calls == [("mem-b-usable", True)]
        assert service.call_order == [
            "retrieve",
            "get:mem-b-usable",
            "write",
            "mark_used",
        ]

        historical = service.get("mem-b-usable")
        assert historical is not None
        assert historical.reuse_count == 1
        assert historical.success_count == 1
        assert historical.failure_count == 0
        assert historical.last_accessed_at is not None

        current = service.get("mem_task-effective")
        assert current is not None
        assert current.reuse_count == 0
        assert current.success_count == 0
        assert current.failure_count == 0
        assert current.last_accessed_at is None

        _assert_four_agent_chain(trace_path)
        assert CACHED_SENTINEL not in trace_path.read_text(encoding="utf-8")
        assert len(encoded_execute_requests) == 1
        assert CACHED_SENTINEL.encode("utf-8") in encoded_execute_requests[0]
    finally:
        store.close()


def test_multiple_retrieved_refs_use_only_first_compatible_memory(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "multiple-memory"
    unusable = _record(
        "mem-a-unusable",
        payload={"answer": "legacy answer"},
    )
    usable = _usable_record("mem-b-usable")
    _seed(memory_root, [unusable, usable])
    store, service = _recording_service(memory_root)
    executor = RecordingExecutorAgent()
    metrics, _, ctx = _context(
        tmp_path,
        name="multiple",
        use_memory=True,
        memory_service=service,
    )
    try:
        final = Orchestrator(Router(_agents(executor))).run_task(task=TASK, ctx=ctx)
        assert final.result is not None
        assert final.result["answer"] == CACHED_ANSWER

        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 2
        assert snapshot.memory_used == 1
        assert snapshot.memory_effective == 1
        assert snapshot.memory_harmful == 0
        assert service.get_calls[:2] == ["mem-a-unusable", "mem-b-usable"]
        assert service.mark_used_calls == [("mem-b-usable", True)]

        first = service.store.get("mem-a-unusable")
        assert first is not None
        assert first.reuse_count == 0
        assert first.success_count == 0
        assert first.failure_count == 0
        assert first.last_accessed_at is None

        second = service.store.get("mem-b-usable")
        assert second is not None
        assert second.reuse_count == 1
        assert second.success_count == 1
        assert second.failure_count == 0
    finally:
        store.close()


def test_retrieved_incompatible_memory_is_not_used_or_marked(tmp_path: Path) -> None:
    memory_root = tmp_path / "retrieved-only-memory"
    _seed(
        memory_root,
        [_record("mem-a-unusable", payload={"answer": "old answer"})],
    )
    store, service = _recording_service(memory_root)
    executor = RecordingExecutorAgent()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="retrieved-only",
        use_memory=True,
        memory_service=service,
    )
    try:
        final = Orchestrator(Router(_agents(executor))).run_task(task=TASK, ctx=ctx)
        assert final.result is not None
        assert final.result["answer"] == BASELINE_ANSWER
        assert _execute_operation(executor)["value"] == {
            "retrieved_document_ids": ["network-manager", "ip-route"]
        }

        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert service.mark_used_calls == []

        historical = service.store.get("mem-a-unusable")
        assert historical is not None
        assert historical.reuse_count == 0
        assert historical.success_count == 0
        assert historical.failure_count == 0
        assert historical.last_accessed_at is None
        assert service.store.get("mem_task-retrieved-only") is not None
        _assert_four_agent_chain(trace_path)
    finally:
        store.close()


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(
            _record(
                "mem-evidence",
                memory_type=MemoryType.EVIDENCE,
                payload={
                    "answer": CACHED_ANSWER,
                    "execution": {
                        "operation": "identity",
                        "output": {"cache_token": CACHED_SENTINEL},
                    },
                },
            ),
            id="evidence-not-result",
        ),
        pytest.param(
            _record(
                "mem-other-task",
                task_topic="different task",
                payload={
                    "answer": CACHED_ANSWER,
                    "execution": {
                        "operation": "identity",
                        "output": {"cache_token": CACHED_SENTINEL},
                    },
                },
            ),
            id="different-task",
        ),
        pytest.param(
            _record(
                "mem-no-answer",
                payload={
                    "execution": {
                        "operation": "identity",
                        "output": {"cache_token": CACHED_SENTINEL},
                    }
                },
            ),
            id="missing-answer",
        ),
        pytest.param(
            _record(
                "mem-empty-answer",
                payload={
                    "answer": "",
                    "execution": {
                        "operation": "identity",
                        "output": {"cache_token": CACHED_SENTINEL},
                    },
                },
            ),
            id="empty-answer",
        ),
        pytest.param(
            _record(
                "mem-non-identity",
                payload={
                    "answer": CACHED_ANSWER,
                    "execution": {"operation": "arithmetic", "output": 2},
                },
            ),
            id="non-identity-execution",
        ),
        pytest.param(
            _record(
                "mem-no-output",
                payload={
                    "answer": CACHED_ANSWER,
                    "execution": {"operation": "identity"},
                },
            ),
            id="missing-output",
        ),
    ],
)
def test_other_retrieved_records_are_not_reusable(
    tmp_path: Path,
    record: MemoryRecord,
) -> None:
    memory_root = tmp_path / record.memory_id
    _seed(memory_root, [record])
    store, service = _recording_service(memory_root)
    executor = RecordingExecutorAgent()
    metrics, _, ctx = _context(
        tmp_path,
        name=record.memory_id,
        use_memory=True,
        memory_service=service,
    )
    try:
        Orchestrator(Router(_agents(executor))).run_task(task=TASK, ctx=ctx)
        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert service.mark_used_calls == []
        assert "retrieved_document_ids" in _execute_operation(executor)["value"]
    finally:
        store.close()


def test_harmful_memory_is_marked_failed_after_current_write(tmp_path: Path) -> None:
    memory_root = tmp_path / "harmful-memory"
    _seed(
        memory_root,
        [
            _usable_record(
                "mem-harmful",
                answer=HISTORICAL_ANSWER,
                sentinel=HARMFUL_SENTINEL,
            )
        ],
    )
    store, service = _recording_service(memory_root)
    executor = RecordingExecutorAgent()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="harmful",
        use_memory=True,
        memory_service=service,
    )
    try:
        final = Orchestrator(Router(_agents(executor))).run_task(task=TASK, ctx=ctx)
        assert final.result is not None
        assert final.result["answer"] == HARMFUL_ANSWER
        assert _execute_operation(executor)["value"] == {
            "cache_token": HARMFUL_SENTINEL
        }

        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 1
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 1
        assert service.mark_used_calls == [("mem-harmful", False)]
        assert service.call_order.index("write") < service.call_order.index("mark_used")

        historical = service.store.get("mem-harmful")
        assert historical is not None
        assert historical.reuse_count == 1
        assert historical.success_count == 0
        assert historical.failure_count == 1
        assert historical.last_accessed_at is not None

        current = service.store.get("mem_task-harmful")
        assert current is not None
        assert current.reuse_count == 0
        assert current.success_count == 0
        assert current.failure_count == 0
        assert current.last_accessed_at is None
        _assert_four_agent_chain(trace_path)
    finally:
        store.close()


def test_missing_resolved_record_is_consistency_error(tmp_path: Path) -> None:
    store = SQLiteMemoryStore(tmp_path / "missing-record")
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = MissingRecordMemoryService(store, provider, VectorIndex(provider.dim))
    executor = RecordingExecutorAgent()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="missing-record",
        use_memory=True,
        memory_service=service,
    )
    try:
        with pytest.raises(ValueError, match="missing record"):
            Orchestrator(Router(_agents(executor))).run_task(task=TASK, ctx=ctx)

        snapshot = metrics.snapshot()
        assert snapshot.message_count == 2
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert service.get_calls == ["mem-missing"]
        assert service.write_calls == 0
        assert service.mark_used_calls == []
        assert len(_events(trace_path)) == 2
    finally:
        store.close()


def test_write_failure_after_actual_use_does_not_mark_historical_memory(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "write-failure-memory"
    _seed(memory_root, [_usable_record()])
    store, service = _recording_service(
        memory_root,
        service_type=FailingWriteMemoryService,
    )
    executor = RecordingExecutorAgent()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="write-failure",
        use_memory=True,
        memory_service=service,
    )
    try:
        with pytest.raises(RuntimeError, match="memory write failed"):
            Orchestrator(Router(_agents(executor))).run_task(task=TASK, ctx=ctx)

        assert _execute_operation(executor)["value"] == {
            "cache_token": CACHED_SENTINEL
        }
        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 1
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert service.mark_used_calls == []
        assert service.store.get("mem_task-write-failure") is None

        historical = service.store.get("mem-b-usable")
        assert historical is not None
        assert historical.reuse_count == 0
        assert historical.success_count == 0
        assert historical.failure_count == 0
        assert historical.last_accessed_at is None
        _assert_four_agent_chain(trace_path)
    finally:
        store.close()


def test_mark_used_failure_propagates_without_feedback_metric_or_rollback(
    tmp_path: Path,
) -> None:
    memory_root = tmp_path / "mark-failure-memory"
    _seed(memory_root, [_usable_record()])
    store, service = _recording_service(
        memory_root,
        service_type=FailingMarkUsedMemoryService,
    )
    executor = RecordingExecutorAgent()
    metrics, trace_path, ctx = _context(
        tmp_path,
        name="mark-failure",
        use_memory=True,
        memory_service=service,
    )
    try:
        with pytest.raises(RuntimeError, match="memory mark_used failed"):
            Orchestrator(Router(_agents(executor))).run_task(task=TASK, ctx=ctx)

        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 1
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert service.mark_used_calls == [("mem-b-usable", True)]
        assert service.call_order.index("write") < service.call_order.index("mark_used")

        historical = service.store.get("mem-b-usable")
        assert historical is not None
        assert historical.reuse_count == 0
        assert historical.success_count == 0
        assert historical.failure_count == 0
        assert historical.last_accessed_at is None

        current = service.store.get("mem_task-mark-failure")
        assert current is not None
        assert current.reuse_count == 0
        assert current.success_count == 0
        assert current.failure_count == 0
        assert current.last_accessed_at is None
        _assert_four_agent_chain(trace_path)
    finally:
        store.close()
