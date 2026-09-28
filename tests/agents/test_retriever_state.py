from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from agentipc.agents.retriever import RetrieverAgent
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageType
from agentipc.providers.hash_embedding import (
    DEFAULT_HASH_EMBEDDING_DIM,
    HashEmbeddingProvider,
)
from agentipc.state.hub import StateHub
from agentipc.state.plan_vector import (
    PLAN_VECTOR_DIM,
    PLAN_VECTOR_KIND,
    encode_plan_vector,
)


class ForbiddenService:
    def __getattr__(self, name: str):
        raise AssertionError(f"Retriever must not access service: {name}")


class RecordingStateHub:
    def __init__(self, delegate: StateHub) -> None:
        self.delegate = delegate
        self.resolve_count = 0
        self.resolved_refs = []

    def resolve_array(self, ref):
        self.resolve_count += 1
        self.resolved_refs.append(ref)
        return self.delegate.resolve_array(ref)


class ForbiddenStateHub:
    def __getattr__(self, name: str):
        raise AssertionError(f"baseline must not access state_hub: {name}")


def _profiles():
    profile_a = {
        "task_type": "network_recovery",
        "steps": ["retrieve", "inspect_network", "summarize"],
        "needs_tool": False,
    }
    profile_b = {
        "task_type": "filesystem_recovery",
        "steps": ["retrieve", "inspect_filesystem", "summarize"],
        "needs_tool": True,
    }
    return profile_a, profile_b


def _knowledge():
    profile_a, profile_b = _profiles()
    return [
        {
            "document_id": "doc-a",
            "text": "shared diagnostic evidence",
            "keywords": ["diagnostic", "evidence"],
            "state_profile": profile_a,
        },
        {
            "document_id": "doc-b",
            "text": "shared diagnostic evidence",
            "keywords": ["diagnostic", "evidence"],
            "state_profile": profile_b,
        },
    ]


def _request(*, state_refs=None):
    return AgentEnvelope(
        message_id="msg-state",
        trace_id="trace-state",
        task_id="task-state",
        step_id="step-state",
        sender="runtime",
        receiver="retriever",
        message_type=MessageType.REQUEST,
        action=ActionType.RETRIEVE,
        args={
            "plan": {
                "task": "diagnostic evidence",
                "retrieval_topics": ["diagnostic", "evidence"],
            }
        },
        state_refs=[] if state_refs is None else state_refs,
    )


def _context(state_hub):
    return SimpleNamespace(
        provider_bundle=SimpleNamespace(
            embedding=HashEmbeddingProvider(),
            llm=ForbiddenService(),
        ),
        state_hub=state_hub,
        memory_service=ForbiddenService(),
        artifact_store=ForbiddenService(),
        metrics=ForbiddenService(),
        trace_logger=ForbiddenService(),
    )


def _ids(output):
    return [item["document_id"] for item in output.result["evidence"]]


def test_real_state_ref_changes_ranking_and_spaces_remain_separate() -> None:
    profile_a, profile_b = _profiles()
    vector_a = encode_plan_vector(profile_a)
    vector_b = encode_plan_vector(profile_b)

    assert vector_a.shape == (PLAN_VECTOR_DIM,)
    assert vector_b.shape == (PLAN_VECTOR_DIM,)
    assert vector_a.dtype == np.float32
    assert vector_b.dtype == np.float32
    assert not np.array_equal(vector_a, vector_b)
    assert DEFAULT_HASH_EMBEDDING_DIM == 128
    assert PLAN_VECTOR_DIM == 64

    knowledge = _knowledge()
    knowledge_before = deepcopy(knowledge)
    agent = RetrieverAgent(knowledge)

    baseline_request = _request()
    baseline_before = baseline_request.model_copy(deep=True)
    baseline = agent.handle(
        baseline_request,
        _context(ForbiddenStateHub()),
    )
    assert _ids(baseline)[:2] == ["doc-a", "doc-b"]
    assert baseline_request == baseline_before

    with StateHub(transport="inproc") as real_hub:
        ref_a = real_hub.put_array(
            vector_a,
            kind=PLAN_VECTOR_KIND,
            summary="planner profile a",
        )
        ref_b = real_hub.put_array(
            vector_b,
            kind=PLAN_VECTOR_KIND,
            summary="planner profile b",
        )
        recording = RecordingStateHub(real_hub)
        ctx = _context(recording)

        request_a = _request(state_refs=[ref_a])
        request_a_before = request_a.model_copy(deep=True)
        output_a = agent.handle(request_a, ctx)

        request_b = _request(state_refs=[ref_b])
        request_b_before = request_b.model_copy(deep=True)
        output_b = agent.handle(request_b, ctx)

        assert _ids(output_a)[0] == "doc-a"
        assert _ids(output_b)[0] == "doc-b"
        assert recording.resolve_count == 2
        assert recording.resolved_refs == [ref_a, ref_b]

        assert output_a.state_refs == []
        assert output_b.state_refs == []
        assert output_a.artifact_refs == []
        assert output_a.memory_refs == []
        assert request_a == request_a_before
        assert request_b == request_b_before
        assert knowledge == knowledge_before

        score_a = output_a.result["evidence"][0]["score"]
        assert score_a > 1.0


def test_non_plan_state_refs_are_ignored_and_not_resolved() -> None:
    with StateHub(transport="inproc") as hub:
        other_ref = hub.put_array(
            np.ones(5, dtype=np.float32),
            kind="other_state",
            summary="ignored",
        )
        output = RetrieverAgent(_knowledge()).handle(
            _request(state_refs=[other_ref]),
            _context(ForbiddenStateHub()),
        )
        assert _ids(output)[:2] == ["doc-a", "doc-b"]
        assert output.state_refs == []


def test_more_than_one_plan_ref_is_rejected_before_resolution() -> None:
    vector = encode_plan_vector(_profiles()[0])
    with StateHub(transport="inproc") as hub:
        first = hub.put_array(vector, kind=PLAN_VECTOR_KIND, summary="first")
        second = hub.put_array(vector, kind=PLAN_VECTOR_KIND, summary="second")
        with pytest.raises(ValueError, match="at most one"):
            RetrieverAgent(_knowledge()).handle(
                _request(state_refs=[first, second]),
                _context(ForbiddenStateHub()),
            )


def test_wrong_plan_vector_dimension_is_rejected() -> None:
    with StateHub(transport="inproc") as hub:
        ref = hub.put_array(
            np.ones(32, dtype=np.float32),
            kind=PLAN_VECTOR_KIND,
            summary="wrong dimension",
        )
        with pytest.raises(ValueError, match="shape"):
            RetrieverAgent(_knowledge()).handle(
                _request(state_refs=[ref]),
                _context(hub),
            )


def test_wrong_plan_vector_dtype_is_rejected_without_casting() -> None:
    with StateHub(transport="inproc") as hub:
        ref = hub.put_array(
            np.ones(PLAN_VECTOR_DIM, dtype=np.int32),
            kind=PLAN_VECTOR_KIND,
            summary="wrong dtype",
        )
        with pytest.raises(ValueError, match="dtype"):
            RetrieverAgent(_knowledge()).handle(
                _request(state_refs=[ref]),
                _context(hub),
            )


def test_zero_norm_plan_vector_is_rejected() -> None:
    with StateHub(transport="inproc") as hub:
        ref = hub.put_array(
            np.zeros(PLAN_VECTOR_DIM, dtype=np.float32),
            kind=PLAN_VECTOR_KIND,
            summary="zero",
        )
        with pytest.raises(ValueError, match="non-zero"):
            RetrieverAgent(_knowledge()).handle(
                _request(state_refs=[ref]),
                _context(hub),
            )


def test_state_profile_schema_is_validated_by_plan_encoder() -> None:
    with pytest.raises(TypeError):
        RetrieverAgent(
            [
                {
                    "document_id": "bad",
                    "text": "x",
                    "state_profile": {"unsupported": object()},
                }
            ]
        )

    with pytest.raises(TypeError):
        RetrieverAgent(
            [
                {
                    "document_id": "bad",
                    "text": "x",
                    "state_profile": ["not", "dict"],
                }
            ]
        )
