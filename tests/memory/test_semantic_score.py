import math

import numpy as np
import pytest

from agentipc.memory.scoring import semantic_scores
from agentipc.memory.vector_index import VectorIndex
from agentipc.providers.hash_embedding import HashEmbeddingProvider


class FixedEmbeddingProvider:
    def __init__(self, vector: np.ndarray, *, dim: int) -> None:
        self._vector = vector
        self._dim = dim
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return self._vector


class NonArrayEmbeddingProvider:
    @property
    def dim(self) -> int:
        return 2

    def embed(self, texts: list[str]):
        return [[1.0, 0.0]]


class WrongShapeEmbeddingProvider:
    @property
    def dim(self) -> int:
        return 2

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


def test_hash_embedding_and_vector_index_real_integration() -> None:
    provider = HashEmbeddingProvider(dim=128)
    index = VectorIndex(dim=provider.dim)

    vectors = provider.embed([
        "linux shared memory state transfer",
        "banana telescope orchestra",
    ])
    index.add("mem-related", vectors[0])
    index.add("mem-unrelated", vectors[1])

    scores = semantic_scores(
        "linux shared memory state",
        embedding_provider=provider,
        vector_index=index,
    )

    assert set(scores) == {"mem-related", "mem-unrelated"}
    assert scores["mem-related"] > scores["mem-unrelated"]
    assert all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores.values())


def test_returns_all_indexed_memories_not_default_top_five() -> None:
    provider = HashEmbeddingProvider(dim=128)
    index = VectorIndex(dim=provider.dim)
    texts = [f"linux memory item {i}" for i in range(7)]
    vectors = provider.embed(texts)
    for i, vector in enumerate(vectors):
        index.add(f"mem-{i}", vector)

    scores = semantic_scores(
        "linux memory",
        embedding_provider=provider,
        vector_index=index,
    )

    assert len(scores) == 7
    assert set(scores) == {f"mem-{i}" for i in range(7)}


def test_empty_index_returns_empty_dict_without_embedding_call() -> None:
    provider = FixedEmbeddingProvider(np.array([[1.0, 0.0]], dtype=np.float32), dim=2)
    index = VectorIndex(2)

    assert semantic_scores("query", embedding_provider=provider, vector_index=index) == {}
    assert provider.calls == []


def test_dimension_mismatch_raises_before_embedding() -> None:
    provider = FixedEmbeddingProvider(np.zeros((1, 64), dtype=np.float32), dim=64)
    index = VectorIndex(128)
    index.add("mem", np.zeros(128, dtype=np.float32))

    with pytest.raises(ValueError, match="dimension"):
        semantic_scores("query", embedding_provider=provider, vector_index=index)
    assert provider.calls == []


def test_empty_query_with_real_hash_embedding_scores_zero() -> None:
    provider = HashEmbeddingProvider(dim=128)
    index = VectorIndex(provider.dim)
    vectors = provider.embed(["linux memory", "banana telescope"])
    index.add("a", vectors[0])
    index.add("b", vectors[1])

    scores = semantic_scores("", embedding_provider=provider, vector_index=index)

    assert scores == {"a": 0.0, "b": 0.0}


def test_real_hash_scores_are_finite_and_in_unit_interval() -> None:
    provider = HashEmbeddingProvider(dim=128)
    index = VectorIndex(provider.dim)
    vectors = provider.embed(["linux shared memory", "python code", "banana"])
    for memory_id, vector in zip(["a", "b", "c"], vectors, strict=True):
        index.add(memory_id, vector)

    scores = semantic_scores("linux memory", embedding_provider=provider, vector_index=index)

    assert all(isinstance(score, float) for score in scores.values())
    assert all(math.isfinite(score) for score in scores.values())
    assert all(0.0 <= score <= 1.0 for score in scores.values())


def test_negative_cosine_is_clamped_to_zero_with_real_vector_index() -> None:
    provider = FixedEmbeddingProvider(np.array([[-1.0, 0.0]], dtype=np.float32), dim=2)
    index = VectorIndex(2)
    index.add("opposite", np.array([1.0, 0.0], dtype=np.float32))

    raw = index.search(np.array([-1.0, 0.0], dtype=np.float32), top_k=1)
    assert raw[0][1] == pytest.approx(-1.0)

    scores = semantic_scores("query", embedding_provider=provider, vector_index=index)
    assert scores == {"opposite": 0.0}
    assert provider.calls == [["query"]]


def test_semantic_scorer_preserves_vector_index_result_order() -> None:
    provider = FixedEmbeddingProvider(np.array([[1.0, 0.0]], dtype=np.float32), dim=2)
    index = VectorIndex(2)
    index.add("b", np.array([0.0, 1.0], dtype=np.float32))
    index.add("a", np.array([0.0, 1.0], dtype=np.float32))
    index.add("top", np.array([1.0, 0.0], dtype=np.float32))

    scores = semantic_scores("query", embedding_provider=provider, vector_index=index)

    assert list(scores) == ["top", "a", "b"]


def test_provider_is_called_with_batch_list_api() -> None:
    provider = FixedEmbeddingProvider(np.array([[1.0, 0.0]], dtype=np.float32), dim=2)
    index = VectorIndex(2)
    index.add("mem", np.array([1.0, 0.0], dtype=np.float32))

    semantic_scores("query", embedding_provider=provider, vector_index=index)

    assert provider.calls == [["query"]]


def test_provider_non_ndarray_output_raises_type_error() -> None:
    provider = NonArrayEmbeddingProvider()
    index = VectorIndex(2)
    index.add("mem", np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(TypeError, match="numpy.ndarray"):
        semantic_scores("query", embedding_provider=provider, vector_index=index)


def test_provider_wrong_shape_raises_value_error() -> None:
    provider = WrongShapeEmbeddingProvider()
    index = VectorIndex(2)
    index.add("mem", np.array([1.0, 0.0], dtype=np.float32))

    with pytest.raises(ValueError, match="shape"):
        semantic_scores("query", embedding_provider=provider, vector_index=index)


@pytest.mark.parametrize("query", [None, 1, b"query", ["query"]])
def test_non_string_query_raises_type_error(query) -> None:
    provider = HashEmbeddingProvider(dim=2)
    index = VectorIndex(2)
    with pytest.raises(TypeError):
        semantic_scores(query, embedding_provider=provider, vector_index=index)


def test_invalid_embedding_provider_raises_type_error() -> None:
    with pytest.raises(TypeError):
        semantic_scores("query", embedding_provider=object(), vector_index=VectorIndex(2))


def test_invalid_vector_index_raises_type_error() -> None:
    provider = HashEmbeddingProvider(dim=2)
    with pytest.raises(TypeError):
        semantic_scores("query", embedding_provider=provider, vector_index=object())
