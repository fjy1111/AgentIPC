import numpy as np
import pytest

from agentipc.memory.vector_index import VectorIndex


def test_constructor_and_initial_length() -> None:
    index = VectorIndex(2)
    assert index.dim == 2
    assert len(index) == 0


@pytest.mark.parametrize("dim", [True, False, 2.0, "2", None])
def test_invalid_dim_type(dim) -> None:
    with pytest.raises(TypeError):
        VectorIndex(dim)


@pytest.mark.parametrize("dim", [0, -1, -10])
def test_non_positive_dim(dim: int) -> None:
    with pytest.raises(ValueError):
        VectorIndex(dim)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([3.0, 4.0], dtype=np.float32),
        np.array([3.0, 4.0], dtype=np.float64),
        np.array([3, 4], dtype=np.int64),
    ],
)
def test_add_accepts_real_numeric_arrays_and_normalizes(vector: np.ndarray) -> None:
    index = VectorIndex(2)
    index.add("mem", vector)
    stored = index._vectors["mem"]
    assert stored.dtype == np.float32
    assert np.linalg.norm(stored) == pytest.approx(1.0)


@pytest.mark.parametrize("memory_id", [None, 1, b"mem"])
def test_add_rejects_non_string_memory_id(memory_id) -> None:
    with pytest.raises(TypeError):
        VectorIndex(2).add(memory_id, np.array([1.0, 0.0]))


def test_add_rejects_empty_memory_id() -> None:
    with pytest.raises(ValueError):
        VectorIndex(2).add("", np.array([1.0, 0.0]))


@pytest.mark.parametrize("vector", [[1.0, 0.0], (1.0, 0.0)])
def test_add_rejects_non_array(vector) -> None:
    with pytest.raises(TypeError):
        VectorIndex(2).add("mem", vector)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([[1.0, 0.0]]),
        np.array([[1.0], [0.0]]),
        np.array([1.0]),
        np.array([1.0, 0.0, 0.0]),
    ],
)
def test_add_rejects_wrong_shape(vector: np.ndarray) -> None:
    with pytest.raises(ValueError):
        VectorIndex(2).add("mem", vector)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([True, False], dtype=bool),
        np.array([1 + 0j, 0 + 1j], dtype=np.complex64),
        np.array([1, 2], dtype=object),
        np.array(["1", "2"]),
    ],
)
def test_add_rejects_non_real_numeric_dtype(vector: np.ndarray) -> None:
    with pytest.raises(TypeError):
        VectorIndex(2).add("mem", vector)


@pytest.mark.parametrize(
    "vector",
    [
        np.array([np.nan, 0.0]),
        np.array([np.inf, 0.0]),
        np.array([-np.inf, 0.0]),
    ],
)
def test_add_rejects_non_finite_values(vector: np.ndarray) -> None:
    with pytest.raises(ValueError):
        VectorIndex(2).add("mem", vector)


def test_add_copies_caller_vector() -> None:
    source = np.array([3.0, 4.0], dtype=np.float32)
    index = VectorIndex(2)
    index.add("mem", source)
    source[:] = 0.0
    assert index.search(np.array([3.0, 4.0], dtype=np.float32), top_k=1)[0][1] == pytest.approx(1.0)


def test_zero_vector_is_accepted_and_scores_zero() -> None:
    index = VectorIndex(2)
    index.add("zero", np.zeros(2, dtype=np.float32))
    assert index.search(np.array([1.0, 0.0], dtype=np.float32)) == [("zero", 0.0)]


def test_duplicate_memory_id_replaces_without_growing() -> None:
    index = VectorIndex(2)
    index.add("mem", np.array([1.0, 0.0]))
    index.add("mem", np.array([0.0, 1.0]))
    assert len(index) == 1
    assert index.search(np.array([0.0, 1.0]), top_k=1)[0] == pytest.approx(("mem", 1.0))


def test_clear_empties_index_and_preserves_dim() -> None:
    index = VectorIndex(2)
    index.add("mem", np.array([1.0, 0.0]))
    index.clear()
    assert len(index) == 0
    assert index.dim == 2


def test_known_cosine_ranking_and_scores() -> None:
    index = VectorIndex(2)
    index.add("same", np.array([1.0, 0.0]))
    index.add("orthogonal", np.array([0.0, 1.0]))
    index.add("opposite", np.array([-1.0, 0.0]))

    results = index.search(np.array([1.0, 0.0]), top_k=3)
    assert [item[0] for item in results] == ["same", "orthogonal", "opposite"]
    assert results[0][1] == pytest.approx(1.0)
    assert results[1][1] == pytest.approx(0.0)
    assert results[2][1] == pytest.approx(-1.0)


def test_top_k_one_and_larger_than_index() -> None:
    index = VectorIndex(2)
    index.add("a", np.array([1.0, 0.0]))
    index.add("b", np.array([0.0, 1.0]))
    assert len(index.search(np.array([1.0, 0.0]), top_k=1)) == 1
    assert len(index.search(np.array([1.0, 0.0]), top_k=99)) == 2


def test_empty_index_returns_empty_list() -> None:
    assert VectorIndex(2).search(np.array([1.0, 0.0])) == []


@pytest.mark.parametrize("query", [[1.0, 0.0], (1.0, 0.0)])
def test_search_rejects_non_array_query(query) -> None:
    with pytest.raises(TypeError):
        VectorIndex(2).search(query)


@pytest.mark.parametrize("top_k", [True, False, 1.0, "1", None])
def test_search_rejects_invalid_top_k_type(top_k) -> None:
    with pytest.raises(TypeError):
        VectorIndex(2).search(np.array([1.0, 0.0]), top_k=top_k)


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_rejects_non_positive_top_k(top_k: int) -> None:
    with pytest.raises(ValueError):
        VectorIndex(2).search(np.array([1.0, 0.0]), top_k=top_k)


def test_zero_query_gives_all_zero_scores_with_id_tie_break() -> None:
    index = VectorIndex(2)
    index.add("b", np.array([1.0, 0.0]))
    index.add("a", np.array([0.0, 1.0]))
    assert index.search(np.zeros(2), top_k=2) == [("a", 0.0), ("b", 0.0)]


def test_ties_sort_by_memory_id_and_ignore_insertion_order() -> None:
    query = np.array([1.0, 0.0])
    vector = np.array([0.0, 1.0])

    first = VectorIndex(2)
    first.add("b", vector)
    first.add("a", vector)

    second = VectorIndex(2)
    second.add("a", vector)
    second.add("b", vector)

    assert first.search(query, top_k=2) == second.search(query, top_k=2)
    assert [item[0] for item in first.search(query, top_k=2)] == ["a", "b"]


def test_large_float64_vector_is_normalized_before_float32_storage() -> None:
    index = VectorIndex(2)
    index.add("large", np.array([1e200, 1e200], dtype=np.float64))
    stored = index._vectors["large"]
    assert np.all(np.isfinite(stored))
    assert np.linalg.norm(stored) == pytest.approx(1.0)
