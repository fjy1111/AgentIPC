from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector, MetricsSnapshot
from agentipc.evaluation.text_counter import TextCounter
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.codec import encode as protocol_encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.bootstrap import run_handshake
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router
from agentipc.runtime.text_transport import TextTransport
from agentipc.state.hub import StateHub
from agentipc.state.plan_vector import encode_plan_vector


TASK = "diagnose openEuler network connectivity"
EXPECTED_ANSWER = "Network troubleshooting completed."
EMBEDDING_DIM = 64

KNOWLEDGE = [
    {
        "document_id": "network-manager",
        "text": (
            "NetworkManager manages openEuler network connections "
            "and connectivity."
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

EXPECTED_BUSINESS_CHAIN = [
    (MessageType.REQUEST.value, ActionType.PLAN.value),
    (MessageType.RESULT.value, ActionType.PLAN.value),
    (MessageType.REQUEST.value, ActionType.RETRIEVE.value),
    (MessageType.RESULT.value, ActionType.RETRIEVE.value),
    (MessageType.REQUEST.value, ActionType.EXECUTE.value),
    (MessageType.RESULT.value, ActionType.EXECUTE.value),
    (MessageType.REQUEST.value, ActionType.SUMMARIZE.value),
    (MessageType.RESULT.value, ActionType.SUMMARIZE.value),
]

EXPECTED_CAPABILITIES = {
    "planner": ["plan"],
    "retriever": ["retrieve"],
    "executor": ["execute"],
    "summarizer": ["summarize"],
}


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"forbidden infrastructure access: {name}")


class RecordingExecutorAgent(ExecutorAgent):
    def __init__(self) -> None:
        self.requests: list[AgentEnvelope] = []

    def handle(self, envelope: AgentEnvelope, ctx: RunContext) -> AgentEnvelope:
        self.requests.append(envelope.model_copy(deep=True))
        return super().handle(envelope, ctx)


def _build_agents(
    executor: ExecutorAgent | None = None,
) -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent() if executor is None else executor,
        SummarizerAgent(),
    ]:
        registry.register(agent)
    return registry


def _build_provider_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={
                "network": EXPECTED_ANSWER,
            },
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(dim=EMBEDDING_DIM),
    )


def _build_memory_service(
    root: Path,
) -> tuple[SQLiteMemoryStore, MemoryService]:
    store = SQLiteMemoryStore(root)
    embedding = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    service = MemoryService(
        store,
        embedding,
        VectorIndex(embedding.dim),
    )
    return store, service


def _build_context(
    tmp_path: Path,
    *,
    name: str,
    mode: RunMode,
    capability_registry: CapabilityRegistry,
    metrics: MetricsCollector,
    state_hub: object | None = None,
    artifact_store: object | None = None,
    memory_service: object | None = None,
    use_state: bool = False,
    use_memory: bool = False,
) -> tuple[TraceLogger, Path, RunContext]:
    trace_path = tmp_path / f"{name}.jsonl"
    trace_logger = TraceLogger(trace_path)
    poison = PoisonService()
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=mode,
        config=AgentIPCConfig(),
        registry=capability_registry,
        state_hub=(poison if state_hub is None else state_hub),  # type: ignore[arg-type]
        artifact_store=(
            poison if artifact_store is None else artifact_store
        ),  # type: ignore[arg-type]
        memory_service=(
            poison if memory_service is None else memory_service
        ),  # type: ignore[arg-type]
        metrics=metrics,
        trace_logger=trace_logger,
        provider_bundle=_build_provider_bundle(),
        use_state=use_state,
        use_memory=use_memory,
        use_sandbox=False,
    )
    return trace_logger, trace_path, ctx


def _read_events(path: Path) -> list[TraceEvent]:
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
        for event in _read_events(path)
        if event.message_type
        in {
            MessageType.REQUEST.value,
            MessageType.RESULT.value,
        }
    ]


def _assert_four_agent_chain(path: Path) -> list[TraceEvent]:
    business = _business_events(path)
    assert len(business) == 8
    assert [
        (event.message_type, event.action)
        for event in business
    ] == EXPECTED_BUSINESS_CHAIN

    for request, result in zip(business[::2], business[1::2], strict=True):
        assert request.message_type == MessageType.REQUEST.value
        assert result.message_type == MessageType.RESULT.value
        assert request.action == result.action
        assert request.step_id == result.step_id
        assert request.trace_id == result.trace_id
        assert request.task_id == result.task_id

    return business


def _assert_capabilities_bootstrapped(registry: CapabilityRegistry) -> None:
    for agent_id, expected in EXPECTED_CAPABILITIES.items():
        capability = registry.get(agent_id)
        assert capability is not None
        assert capability.capabilities == expected
        assert registry.supports(agent_id, expected[0]) is True


def _execution(final: AgentEnvelope) -> dict[str, object]:
    assert final.result is not None
    candidate = final.result["memory_candidate"]
    assert type(candidate) is dict
    payload = candidate["payload"]
    assert type(payload) is dict
    execution = payload["execution"]
    assert type(execution) is dict
    return execution


def _assert_no_state_or_memory(snapshot: MetricsSnapshot) -> None:
    assert snapshot.state_transfer_count == 0
    assert snapshot.state_bytes == 0
    assert snapshot.memory_retrieved == 0
    assert snapshot.memory_used == 0
    assert snapshot.memory_effective == 0
    assert snapshot.memory_harmful == 0


def test_text_and_structured_full_mock_tasks_are_business_equivalent(
    tmp_path: Path,
) -> None:
    structured_agents = _build_agents()
    text_agents = _build_agents()
    structured_capabilities = CapabilityRegistry()
    text_capabilities = CapabilityRegistry()
    structured_metrics = MetricsCollector()
    text_metrics = MetricsCollector()

    (
        structured_trace_logger,
        structured_trace_path,
        structured_ctx,
    ) = _build_context(
        tmp_path,
        name="structured-equivalence",
        mode=RunMode.STRUCTURED,
        capability_registry=structured_capabilities,
        metrics=structured_metrics,
    )
    text_trace_logger, text_trace_path, text_ctx = _build_context(
        tmp_path,
        name="text-equivalence",
        mode=RunMode.TEXT,
        capability_registry=text_capabilities,
        metrics=text_metrics,
    )

    assert structured_agents is not text_agents
    assert structured_capabilities is not text_capabilities
    assert structured_metrics is not text_metrics
    assert structured_ctx.provider_bundle is not text_ctx.provider_bundle
    assert structured_ctx.trace_logger is not text_ctx.trace_logger

    structured_handshake = run_handshake(
        agent_registry=structured_agents,
        capability_registry=structured_capabilities,
        trace_logger=structured_trace_logger,
        trace_id=structured_ctx.trace_id,
        task_id=structured_ctx.task_id,
    )
    text_handshake = run_handshake(
        agent_registry=text_agents,
        capability_registry=text_capabilities,
        trace_logger=text_trace_logger,
        trace_id=text_ctx.trace_id,
        task_id=text_ctx.task_id,
    )
    assert len(structured_handshake) == 12
    assert len(text_handshake) == 12
    _assert_capabilities_bootstrapped(structured_capabilities)
    _assert_capabilities_bootstrapped(text_capabilities)

    assert structured_metrics.snapshot().message_count == 0
    assert text_metrics.snapshot().message_count == 0

    structured_final = Orchestrator(
        Router(structured_agents)
    ).run_task(
        task=TASK,
        ctx=structured_ctx,
    )
    structured_snapshot = structured_metrics.snapshot()
    text_before_run = text_metrics.snapshot()
    assert text_before_run == MetricsSnapshot()

    text_final = Orchestrator(
        Router(text_agents),
        text_transport=TextTransport(
            text_counter=TextCounter(use_tiktoken=False),
        ),
    ).run_task(
        task=TASK,
        ctx=text_ctx,
    )
    text_snapshot = text_metrics.snapshot()

    assert structured_metrics.snapshot() == structured_snapshot
    assert structured_final.result == text_final.result
    assert structured_final.message_type is text_final.message_type
    assert structured_final.action is text_final.action
    assert structured_final.status is text_final.status
    assert structured_final.capability == text_final.capability

    for final in (structured_final, text_final):
        assert final.message_type is MessageType.RESULT
        assert final.action is ActionType.SUMMARIZE
        assert final.status is MessageStatus.OK
        assert final.capability == "summarize"
        assert final.result is not None
        assert final.result["answer"] == EXPECTED_ANSWER

    structured_execution = _execution(structured_final)
    text_execution = _execution(text_final)
    assert structured_execution == text_execution
    assert structured_execution["operation"] == "identity"
    structured_ids = structured_execution["output"]["retrieved_document_ids"]
    text_ids = text_execution["output"]["retrieved_document_ids"]
    assert structured_ids == text_ids
    assert "network-manager" in structured_ids

    assert structured_snapshot.message_count == 8
    assert structured_snapshot.protocol_bytes > 0
    assert structured_snapshot.text_chars == 0
    assert structured_snapshot.text_tokens == 0
    assert structured_snapshot.artifact_ref_count == 0
    _assert_no_state_or_memory(structured_snapshot)

    assert text_snapshot.message_count == 8
    assert text_snapshot.text_chars > 0
    assert text_snapshot.text_tokens == 0
    assert text_snapshot.protocol_bytes == 0
    assert text_snapshot.artifact_ref_count == 0
    _assert_no_state_or_memory(text_snapshot)

    structured_business = _assert_four_agent_chain(structured_trace_path)
    text_business = _assert_four_agent_chain(text_trace_path)
    assert len(_read_events(structured_trace_path)) == 20
    assert len(_read_events(text_trace_path)) == 20
    assert [
        (event.message_type, event.action)
        for event in structured_business
    ] == [
        (event.message_type, event.action)
        for event in text_business
    ]


def test_structured_state_artifact_regression_preserves_result_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_hub = StateHub(transport="inproc")
    artifact_store = ArtifactStore(tmp_path / "state-artifact-store")
    preexisting_ref = state_hub.put_array(
        np.array([7.0], dtype=np.float32),
        kind="caller-owned",
        summary="preexisting state",
    )
    try:
        registry = _build_agents()
        capabilities = CapabilityRegistry()
        metrics = MetricsCollector()
        _, trace_path, ctx = _build_context(
            tmp_path,
            name="state-artifact-regression",
            mode=RunMode.STRUCTURED,
            capability_registry=capabilities,
            metrics=metrics,
            state_hub=state_hub,
            artifact_store=artifact_store,
            use_state=True,
            use_memory=False,
        )

        encoded: list[tuple[AgentEnvelope, bytes]] = []

        def recording_encode(envelope: AgentEnvelope) -> bytes:
            payload = protocol_encode(envelope)
            encoded.append((envelope.model_copy(deep=True), payload))
            return payload

        monkeypatch.setattr(
            "agentipc.runtime.orchestrator.encode",
            recording_encode,
        )

        final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
        assert final.result is not None
        assert final.result["answer"] == EXPECTED_ANSWER
        assert final.message_type is MessageType.RESULT
        assert final.action is ActionType.SUMMARIZE
        assert final.status is MessageStatus.OK
        assert final.artifact_refs == []
        assert _execution(final)["operation"] == "identity"
        assert "network-manager" in _execution(final)["output"][
            "retrieved_document_ids"
        ]

        snapshot = metrics.snapshot()
        assert snapshot.message_count == 8
        assert snapshot.protocol_bytes > 0
        assert snapshot.text_chars == 0
        assert snapshot.text_tokens == 0
        assert snapshot.state_transfer_count == 1
        assert snapshot.state_bytes == 256
        assert snapshot.artifact_ref_count == 1
        assert snapshot.memory_retrieved == 0
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0

        business = _assert_four_agent_chain(trace_path)
        state_events = [event for event in business if event.state_refs]
        assert len(state_events) == 1
        state_event = state_events[0]
        assert state_event.message_type == MessageType.REQUEST.value
        assert state_event.action == ActionType.RETRIEVE.value
        assert len(state_event.state_refs) == 1
        plan_state_ref = state_event.state_refs[0]
        assert plan_state_ref.nbytes == 256
        assert sum(len(event.state_refs) for event in business) == 1

        artifact_events = [event for event in business if event.artifact_refs]
        assert len(artifact_events) == 1
        artifact_event = artifact_events[0]
        assert artifact_event.message_type == MessageType.REQUEST.value
        assert artifact_event.action == ActionType.EXECUTE.value
        assert len(artifact_event.artifact_refs) == 1
        artifact_ref = artifact_event.artifact_refs[0]
        assert sum(len(event.artifact_refs) for event in business) == 1

        retrieve_requests = [
            envelope
            for envelope, _ in encoded
            if envelope.message_type is MessageType.REQUEST
            and envelope.action is ActionType.RETRIEVE
        ]
        assert len(retrieve_requests) == 1
        retrieve_request = retrieve_requests[0]
        assert retrieve_request.state_refs == [plan_state_ref]
        assert type(retrieve_request.args["plan"]) is dict

        retrieve_wires = [
            payload
            for envelope, payload in encoded
            if envelope.message_type is MessageType.REQUEST
            and envelope.action is ActionType.RETRIEVE
        ]
        assert len(retrieve_wires) == 1
        retrieve_wire = retrieve_wires[0]
        assert plan_state_ref.uri.encode() in retrieve_wire
        assert b'"state_refs"' in retrieve_wire
        assert b'"values"' not in retrieve_wire
        raw_plan_vector = encode_plan_vector(retrieve_request.args["plan"])
        assert str(raw_plan_vector.tolist()).encode() not in retrieve_wire

        execute_requests = [
            envelope
            for envelope, _ in encoded
            if envelope.message_type is MessageType.REQUEST
            and envelope.action is ActionType.EXECUTE
        ]
        assert len(execute_requests) == 1
        assert execute_requests[0].artifact_refs == [artifact_ref]

        assert state_hub.exists(plan_state_ref) is False
        assert state_hub.exists(preexisting_ref) is True
        assert artifact_store.exists(artifact_ref) is True
    finally:
        state_hub.release(preexisting_ref)
        state_hub.close()


def test_structured_state_memory_warm_reuse_is_effective_and_isolated(
    tmp_path: Path,
) -> None:
    state_hub = StateHub(transport="inproc")
    artifact_store = ArtifactStore(tmp_path / "memory-artifacts")
    store, memory_service = _build_memory_service(tmp_path / "memory-store")
    preexisting_ref = state_hub.put_array(
        np.array([11.0], dtype=np.float32),
        kind="caller-owned",
        summary="shared preexisting state",
    )
    try:
        cold_registry = _build_agents()
        cold_metrics = MetricsCollector()
        cold_capabilities = CapabilityRegistry()
        _, cold_trace_path, cold_ctx = _build_context(
            tmp_path,
            name="memory-cold",
            mode=RunMode.STRUCTURED,
            capability_registry=cold_capabilities,
            metrics=cold_metrics,
            state_hub=state_hub,
            artifact_store=artifact_store,
            memory_service=memory_service,
            use_state=True,
            use_memory=True,
        )

        cold_final = Orchestrator(Router(cold_registry)).run_task(
            task=TASK,
            ctx=cold_ctx,
        )
        assert cold_final.result is not None
        assert cold_final.result["answer"] == EXPECTED_ANSWER
        cold_snapshot = cold_metrics.snapshot()
        assert cold_snapshot.message_count == 8
        assert cold_snapshot.protocol_bytes > 0
        assert cold_snapshot.state_transfer_count == 1
        assert cold_snapshot.state_bytes == 256
        assert cold_snapshot.artifact_ref_count == 1
        assert cold_snapshot.memory_retrieved == 0
        assert cold_snapshot.memory_used == 0
        assert cold_snapshot.memory_effective == 0
        assert cold_snapshot.memory_harmful == 0

        cold_business = _assert_four_agent_chain(cold_trace_path)
        cold_state_events = [event for event in cold_business if event.state_refs]
        assert len(cold_state_events) == 1
        cold_plan_ref = cold_state_events[0].state_refs[0]
        assert state_hub.exists(cold_plan_ref) is False
        assert state_hub.exists(preexisting_ref) is True

        cold_artifact_events = [
            event for event in cold_business if event.artifact_refs
        ]
        assert len(cold_artifact_events) == 1
        cold_artifact_ref = cold_artifact_events[0].artifact_refs[0]
        assert artifact_store.exists(cold_artifact_ref) is True

        historical_id = f"mem_{cold_ctx.task_id}"
        historical_before = memory_service.get(historical_id)
        assert historical_before is not None
        assert historical_before.task_topic == TASK
        assert historical_before.payload["answer"] == EXPECTED_ANSWER
        historical_execution = historical_before.payload["execution"]
        assert type(historical_execution) is dict
        assert historical_execution["operation"] == "identity"
        historical_output = historical_execution["output"]
        assert historical_before.reuse_count == 0
        assert historical_before.success_count == 0
        assert historical_before.failure_count == 0
        assert historical_before.last_accessed_at is None
        assert len(store.list_records()) == 1

        warm_executor = RecordingExecutorAgent()
        warm_registry = _build_agents(warm_executor)
        warm_metrics = MetricsCollector()
        warm_capabilities = CapabilityRegistry()
        _, warm_trace_path, warm_ctx = _build_context(
            tmp_path,
            name="memory-warm",
            mode=RunMode.STRUCTURED,
            capability_registry=warm_capabilities,
            metrics=warm_metrics,
            state_hub=state_hub,
            artifact_store=artifact_store,
            memory_service=memory_service,
            use_state=True,
            use_memory=True,
        )

        assert cold_ctx.task_id != warm_ctx.task_id
        assert cold_ctx.trace_id != warm_ctx.trace_id
        assert cold_ctx.metrics is not warm_ctx.metrics
        assert cold_trace_path != warm_trace_path

        warm_final = Orchestrator(Router(warm_registry)).run_task(
            task=TASK,
            ctx=warm_ctx,
        )
        assert warm_final.result is not None
        assert warm_final.result["answer"] == EXPECTED_ANSWER
        assert warm_final.result["answer"] == historical_before.payload["answer"]

        warm_snapshot = warm_metrics.snapshot()
        assert warm_snapshot.message_count == 8
        assert warm_snapshot.protocol_bytes > 0
        assert warm_snapshot.state_transfer_count == 1
        assert warm_snapshot.state_bytes == 256
        assert warm_snapshot.artifact_ref_count == 1
        assert warm_snapshot.memory_retrieved >= 1
        assert warm_snapshot.memory_used == 1
        assert warm_snapshot.memory_effective == 1
        assert warm_snapshot.memory_harmful == 0
        assert cold_metrics.snapshot() == cold_snapshot

        assert len(warm_executor.requests) == 1
        operation = warm_executor.requests[0].args["operation"]
        assert type(operation) is dict
        assert operation["name"] == "identity"
        assert operation["value"] == historical_output

        warm_business = _assert_four_agent_chain(warm_trace_path)
        warm_state_events = [event for event in warm_business if event.state_refs]
        assert len(warm_state_events) == 1
        warm_plan_ref = warm_state_events[0].state_refs[0]
        assert state_hub.exists(warm_plan_ref) is False
        assert state_hub.exists(preexisting_ref) is True

        warm_artifact_events = [
            event for event in warm_business if event.artifact_refs
        ]
        assert len(warm_artifact_events) == 1
        warm_artifact_ref = warm_artifact_events[0].artifact_refs[0]
        assert artifact_store.exists(cold_artifact_ref) is True
        assert artifact_store.exists(warm_artifact_ref) is True

        historical_after = memory_service.get(historical_id)
        assert historical_after is not None
        assert historical_after.reuse_count == 1
        assert historical_after.success_count == 1
        assert historical_after.failure_count == 0
        assert historical_after.last_accessed_at is not None

        current_id = f"mem_{warm_ctx.task_id}"
        current = memory_service.get(current_id)
        assert current is not None
        assert current.task_topic == TASK
        records = store.list_records()
        assert {record.memory_id for record in records} == {
            historical_id,
            current_id,
        }

        probe_ref = state_hub.put_array(
            np.array([13.0], dtype=np.float32),
            kind="post-run-probe",
            summary="state service remains open",
        )
        assert state_hub.exists(probe_ref) is True
        state_hub.release(probe_ref)
        assert state_hub.exists(probe_ref) is False
    finally:
        store.close()
        state_hub.release(preexisting_ref)
        state_hub.close()
