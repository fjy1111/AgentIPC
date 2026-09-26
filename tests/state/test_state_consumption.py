import numpy as np

from agentipc.state.hub import StateHub
from agentipc.state.plan_vector import (
    PLAN_VECTOR_DIM,
    PLAN_VECTOR_KIND,
    encode_plan_vector,
)


def test_resolved_shm_plan_state_changes_candidate_ranking() -> None:
    planner_plan = {
        "task_type": "log_analysis",
        "steps": [
            "retrieve_evidence",
            "inspect_logs",
            "summarize",
        ],
        "required_capabilities": [
            "retrieve",
            "execute",
            "summarize",
        ],
        "needs_tool": True,
    }
    relevant_profile = {
        "task_type": "log_analysis",
        "steps": [
            "retrieve_evidence",
            "inspect_logs",
            "summarize",
        ],
        "required_capabilities": [
            "retrieve",
            "execute",
            "summarize",
        ],
        "needs_tool": True,
    }
    unrelated_profile = {
        "domain": "weather_forecast",
    }

    baseline_scores = np.array([0.45, 0.55], dtype=np.float32)
    state_weight = np.float32(0.5)

    with StateHub(transport="shm") as hub:
        plan_vector = encode_plan_vector(planner_plan)
        ref = hub.put_array(
            plan_vector,
            kind=PLAN_VECTOR_KIND,
            summary="planner compact state",
        )

        assert ref.shape == [PLAN_VECTOR_DIM]
        assert np.dtype(ref.dtype) == np.dtype(np.float32)
        assert ref.nbytes == PLAN_VECTOR_DIM * np.dtype(np.float32).itemsize
        assert ref.transport == "shm"
        assert ref.kind == PLAN_VECTOR_KIND

        resolved = hub.resolve_array(ref)
        del plan_vector

        assert resolved.shape == (PLAN_VECTOR_DIM,)
        assert resolved.dtype == np.float32

        relevant_profile_vector = encode_plan_vector(relevant_profile)
        unrelated_profile_vector = encode_plan_vector(unrelated_profile)
        state_scores = np.array(
            [
                float(np.dot(resolved, relevant_profile_vector)),
                float(np.dot(resolved, unrelated_profile_vector)),
            ],
            dtype=np.float32,
        )
        final_scores = baseline_scores + state_weight * state_scores

        assert int(np.argmax(baseline_scores)) == 1
        assert state_scores[0] > state_scores[1]
        assert int(np.argmax(final_scores)) == 0