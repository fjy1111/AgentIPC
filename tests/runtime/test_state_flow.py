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
from agentipc.evaluation.text_counter import TextCounter
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.protocol.codec import encode as real_encode
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
from agentipc.runtime.text_transport import TextTransport
from agentipc.state.hub import StateHub
from agentipc.state.plan_vector import (
    PLAN_VECTOR_DIM,
    PLAN_VECTOR_KIND,
    encode_plan_vector,
)


TASK = "diagnostic evidence"
SHARED_TEXT = "shared diagnostic evidence"
PLANNER_NOTE = "State-aware retrieval completed."

EXPECTED_PLAN = {
    "task": TASK,
    "steps": [
        "retrieve",
        "execute",
        "summarize",
    ],
    "required_capabilities": [
        "retrieve",
        "execute",
        "summarize",
    ],
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
    def __init__(self) -> None:
        super().__init__(transport="inproc")
        self.put_calls = 0
        self.resolve_calls = 0
        self.put_arrays: list[np.ndarray] = []
        self.put_refs = []
        self.resolved_refs = []

    def put_array(
        self,
        array: np.ndarray,
        *,
        kind: str,
        summary: str,
    ):
        self.put_calls += 1
        self.put_arrays.append(np.array(array, copy=True))
        ref = super().put_array(array, kind=kind, summary=summary)
        self.put_refs.append(ref)
        return ref

    def resolve_array(self, ref):
        self.resolve_calls += 1
        self.resolved_refs.append(ref)
        return super().resolve_array(ref)


class FailingResolveStateHub(RecordingStateHub):
    def resolve_array(self, ref):
        self.resolve_calls += 1
        self.resolved_refs.append(ref)
        raise RuntimeError("state resolve failed")


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
    name: str,
    use_state: object,
    state_hub: object,
    mode: RunMode = RunMode.STRUCTURED,
    artifact_store: object | None = None,
):
    agent_registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent(),
        SummarizerAgent(),
    ]:
        agent_registry.register(agent)

    metrics = MetricsCollector()
    trace_path = tmp_path / f"{name}.jsonl"
    trace_logger = TraceLogger(trace_path)
    poison = PoisonService()
    ctx = RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=mode,
        config=AgentIPCConfig(),
        registry=CapabilityRegistry(),
        state_hub=state_hub,  # type: ignore[arg-type]
        artifact_store=poison if artifact_store is None else artifact_store,  # type: ignore[arg-type]
        memory_service=poison,  # type: ignore[arg-type]
        metrics=metrics,
        trace_logger=trace_logger,
        provider_bundle=ProviderBundle(
            llm=MockLLMProvider(
                keyword_responses={TASK: PLANNER_NOTE},
                default_text="mock response",
            ),
            embedding=HashEmbeddingProvider(),
        ),
        use_state=use_state,  # type: ignore[arg-type]
        use_memory=False,
        use_sandbox=False,
    )
    return agent_registry, metrics, trace_path, ctx


def _retrieved_document_ids(final: AgentEnvelope) -> list[str]:
    assert final.result is not None
    return final.result["memory_candidate"]["payload"]["execution"]["output"][
        "retrieved_document_ids"
    ]


def test_runtime_state_flow_changes_ranking_and_records_real_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_registry, baseline_metrics, baseline_trace_path, baseline_ctx = _runtime(
        tmp_path,
        name="baseline",
        use_state=False,
        state_hub=PoisonService(),
    )
    baseline_final = Orchestrator(Router(baseline_registry)).run_task(
        task=TASK,
        ctx=baseline_ctx,
    )

    baseline_ids = _retrieved_document_ids(baseline_final)
    assert baseline_ids[0] == "doc-a"
    assert baseline_final.result is not None
    assert baseline_final.result["evidence_summary"].startswith("doc-a:")
    baseline_snapshot = baseline_metrics.snapshot()
    assert baseline_snapshot.message_count == 8
    assert baseline_snapshot.state_transfer_count == 0
    assert baseline_snapshot.state_bytes == 0
    assert baseline_snapshot.artifact_ref_count == 0
    assert all(event.state_refs == [] for event in _events(baseline_trace_path))

    state_hub = RecordingStateHub()
    try:
        state_registry, state_metrics, state_trace_path, state_ctx = _runtime(
            tmp_path,
            name="state",
            use_state=True,
            state_hub=state_hub,
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

        state_final = Orchestrator(Router(state_registry)).run_task(
            task=TASK,
            ctx=state_ctx,
        )

        state_ids = _retrieved_document_ids(state_final)
        assert state_ids[0] == "doc-z"
        assert state_ids != baseline_ids
        assert state_final.result is not None
        assert state_final.result["evidence_summary"].startswith("doc-z:")

        assert state_hub.put_calls == 1
        assert state_hub.resolve_calls == 1
        assert len(state_hub.put_arrays) == 1
        assert len(state_hub.put_refs) == 1
        assert len(state_hub.resolved_refs) == 1

        expected_vector = encode_plan_vector(EXPECTED_PLAN)
        recorded_array = state_hub.put_arrays[0]
        assert np.array_equal(recorded_array, expected_vector)
        assert recorded_array.dtype == np.dtype(np.float32)
        assert recorded_array.shape == (PLAN_VECTOR_DIM,)
        assert bool(np.all(np.isfinite(recorded_array)))
        assert float(np.linalg.norm(recorded_array)) > 0.0

        ref = state_hub.put_refs[0]
        resolved_ref = state_hub.resolved_refs[0]
        assert resolved_ref.uri == ref.uri
        assert resolved_ref.checksum == ref.checksum
        assert resolved_ref.kind == ref.kind
        assert ref.kind == PLAN_VECTOR_KIND
        assert ref.shape == [PLAN_VECTOR_DIM]
        assert ref.dtype == np.dtype(np.float32).str
        assert ref.nbytes == PLAN_VECTOR_DIM * np.dtype(np.float32).itemsize == 256
        assert ref.summary == "planner plan vector"
        assert ref.transport in {"inproc", "shm"}
        assert ref.checksum
        assert ref.uri.startswith(("inproc://agentipc/", "shm://agentipc/"))
        assert state_hub.exists(ref) is True

        snapshot = state_metrics.snapshot()
        assert snapshot.message_count == 8
        assert snapshot.protocol_bytes > 0
        assert snapshot.state_transfer_count == 1
        assert snapshot.state_bytes == ref.nbytes == 256
        assert snapshot.artifact_ref_count == 0
        assert snapshot.memory_retrieved == 0
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert snapshot.latency_ms == 0.0
        assert snapshot.success is False

        business_events = _events(state_trace_path)
        assert len(business_events) == 8
        state_events = [event for event in business_events if event.state_refs]
        assert len(state_events) == 1
        state_event = state_events[0]
        assert state_event.message_type == MessageType.REQUEST.value
        assert state_event.action == ActionType.RETRIEVE.value
        assert len(state_event.state_refs) == 1
        assert state_event.state_refs[0] == ref
        assert sum(len(event.state_refs) for event in business_events) == 1

        retrieve_wire = [
            payload
            for envelope, payload in encoded
            if envelope.message_type is MessageType.REQUEST
            and envelope.action is ActionType.RETRIEVE
        ]
        assert len(retrieve_wire) == 1
        wire = retrieve_wire[0]
        assert b'"state_refs"' in wire
        assert PLAN_VECTOR_KIND.encode() in wire
        assert b"planner plan vector" in wire
        assert ref.uri.encode() in wire
        assert b'"values"' not in wire
        assert str(recorded_array.tolist()).encode() not in wire
    finally:
        state_hub.close()



def test_state_and_artifact_paths_compose(tmp_path: Path) -> None:
    state_hub = RecordingStateHub()
    artifact_store = ArtifactStore(tmp_path / "combined-artifacts")
    try:
        registry, metrics, _, ctx = _runtime(
            tmp_path,
            name="state-artifact-compose",
            use_state=True,
            state_hub=state_hub,
            artifact_store=artifact_store,
        )
        final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
        assert _retrieved_document_ids(final)[0] == "doc-z"
        snapshot = metrics.snapshot()
        assert snapshot.message_count == 8
        assert snapshot.state_transfer_count == 1
        assert snapshot.state_bytes == 256
        assert snapshot.artifact_ref_count == 1
        assert state_hub.put_calls == 1
        assert state_hub.resolve_calls == 1
    finally:
        state_hub.close()

def test_use_state_false_never_accesses_state_hub(tmp_path: Path) -> None:
    registry, metrics, _, ctx = _runtime(
        tmp_path,
        name="disabled-poison",
        use_state=False,
        state_hub=PoisonService(),
    )
    final = Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
    assert _retrieved_document_ids(final)[0] == "doc-a"
    snapshot = metrics.snapshot()
    assert snapshot.state_transfer_count == 0
    assert snapshot.state_bytes == 0


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_use_state_requires_exact_bool_before_dispatch(
    tmp_path: Path,
    value: object,
) -> None:
    registry, metrics, trace_path, ctx = _runtime(
        tmp_path,
        name=f"invalid-bool-{type(value).__name__}-{value}",
        use_state=value,
        state_hub=PoisonService(),
    )
    with pytest.raises(TypeError, match="use_state.*bool"):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
    assert metrics.snapshot().message_count == 0
    assert _events(trace_path) == []


def test_state_enabled_requires_real_state_hub_before_dispatch(tmp_path: Path) -> None:
    registry, metrics, trace_path, ctx = _runtime(
        tmp_path,
        name="invalid-hub",
        use_state=True,
        state_hub=PoisonService(),
    )
    with pytest.raises(
        TypeError,
        match="use_state=True requires ctx.state_hub to be a StateHub",
    ):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
    assert metrics.snapshot().message_count == 0
    assert _events(trace_path) == []


def test_text_mode_with_state_is_rejected_before_dispatch(tmp_path: Path) -> None:
    state_hub = RecordingStateHub()
    try:
        registry, metrics, trace_path, ctx = _runtime(
            tmp_path,
            name="text-state",
            use_state=True,
            state_hub=state_hub,
            mode=RunMode.TEXT,
        )
        orchestrator = Orchestrator(
            Router(registry),
            text_transport=TextTransport(
                text_counter=TextCounter(use_tiktoken=False),
            ),
        )
        with pytest.raises(ValueError, match="does not support use_state=True"):
            orchestrator.run_task(task=TASK, ctx=ctx)
        assert metrics.snapshot().message_count == 0
        assert _events(trace_path) == []
        assert state_hub.put_calls == 0
        assert state_hub.resolve_calls == 0
    finally:
        state_hub.close()


def test_state_resolve_failure_propagates_without_baseline_fallback(
    tmp_path: Path,
) -> None:
    state_hub = FailingResolveStateHub()
    try:
        registry, metrics, trace_path, ctx = _runtime(
            tmp_path,
            name="resolve-failure",
            use_state=True,
            state_hub=state_hub,
        )
        with pytest.raises(RuntimeError, match="state resolve failed"):
            Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)

        assert state_hub.put_calls == 1
        assert state_hub.resolve_calls == 1
        assert len(state_hub.put_refs) == 1
        assert state_hub.exists(state_hub.put_refs[0]) is True
        snapshot = metrics.snapshot()
        assert snapshot.message_count == 3
        assert snapshot.state_transfer_count == 1
        assert snapshot.state_bytes == 256
        events = _events(trace_path)
        assert [(event.message_type, event.action) for event in events] == [
            (MessageType.REQUEST.value, ActionType.PLAN.value),
            (MessageType.RESULT.value, ActionType.PLAN.value),
            (MessageType.REQUEST.value, ActionType.RETRIEVE.value),
        ]
    finally:
        state_hub.close()
