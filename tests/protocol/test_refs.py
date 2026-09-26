import math

import pytest
from pydantic import ValidationError

from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef


def make_state_ref(**overrides) -> StateRef:
    data = {
        "uri": "shm://agentipc/plan-state",
        "kind": "plan_embedding",
        "shape": [4, 8],
        "dtype": "float32",
        "nbytes": 128,
        "checksum": "checksum-value",
        "transport": "shm",
        "summary": "planner state",
    }
    data.update(overrides)
    return StateRef(**data)


def make_artifact_ref(**overrides) -> ArtifactRef:
    data = {
        "uri": "artifact://sha256/example",
        "sha256": "a" * 64,
        "media_type": "application/json",
        "size_bytes": 42,
        "summary": "retrieval evidence",
    }
    data.update(overrides)
    return ArtifactRef(**data)


def make_memory_ref(**overrides) -> MemoryRef:
    data = {
        "memory_id": "mem_example",
        "score": 0.75,
        "match_type": "semantic",
        "summary": "reusable prior result",
    }
    data.update(overrides)
    return MemoryRef(**data)


def test_state_ref_constructs_and_round_trips() -> None:
    ref = make_state_ref()

    restored = StateRef.model_validate(ref.model_dump())

    assert restored == ref


def test_state_ref_rejects_negative_nbytes() -> None:
    with pytest.raises(ValidationError):
        make_state_ref(nbytes=-1)


def test_state_ref_rejects_negative_shape_dimension() -> None:
    with pytest.raises(ValidationError):
        make_state_ref(shape=[4, -1])


def test_state_ref_rejects_unknown_extra_field() -> None:
    with pytest.raises(ValidationError):
        make_state_ref(unexpected=True)


def test_artifact_ref_accepts_valid_sha256_and_nonnegative_size() -> None:
    ref = make_artifact_ref()

    assert len(ref.sha256) == 64
    assert ref.size_bytes == 42


def test_artifact_ref_rejects_negative_size() -> None:
    with pytest.raises(ValidationError):
        make_artifact_ref(size_bytes=-1)


def test_artifact_ref_rejects_invalid_sha256() -> None:
    with pytest.raises(ValidationError):
        make_artifact_ref(sha256="not-a-sha256")


def test_artifact_ref_rejects_unknown_extra_field() -> None:
    with pytest.raises(ValidationError):
        make_artifact_ref(unexpected=True)


def test_memory_ref_constructs_with_finite_score() -> None:
    ref = make_memory_ref(score=1.25)

    assert math.isfinite(ref.score)


@pytest.mark.parametrize(
    "score",
    [
        float("inf"),
        float("-inf"),
        float("nan"),
    ],
)
def test_memory_ref_rejects_non_finite_score(score: float) -> None:
    with pytest.raises(ValidationError):
        make_memory_ref(score=score)


def test_memory_ref_rejects_unknown_extra_field() -> None:
    with pytest.raises(ValidationError):
        make_memory_ref(unexpected=True)