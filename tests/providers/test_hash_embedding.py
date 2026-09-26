import unicodedata

import numpy as np
import pytest

from agentipc.providers.hash_embedding import (
    DEFAULT_HASH_EMBEDDING_DIM,
    HashEmbeddingProvider,
)


def test_default_dim_is_128() -> None:
    provider = HashEmbeddingProvider()

    assert DEFAULT_HASH_EMBEDDING_DIM == 128
    assert provider.dim == 128


def test_custom_positive_dim_works() -> None:
    provider = HashEmbeddingProvider(dim=17)

    assert provider.dim == 17
    assert provider.embed(["hello"]).shape == (1, 17)


@pytest.mark.parametrize("dim", ["128", 128.0, None])
def test_invalid_dim_type_is_rejected(dim: object) -> None:
    with pytest.raises(TypeError):
        HashEmbeddingProvider(dim=dim)  # type: ignore[arg-type]


def test_bool_dim_is_rejected() -> None:
    with pytest.raises(TypeError):
        HashEmbeddingProvider(dim=True)


@pytest.mark.parametrize("dim", [0, -1])
def test_non_positive_dim_is_rejected(dim: int) -> None:
    with pytest.raises(ValueError):
        HashEmbeddingProvider(dim=dim)


def test_same_text_produces_exactly_same_vector() -> None:
    provider = HashEmbeddingProvider()

    first = provider.embed(["Linux shared memory"])
    second = provider.embed(["Linux shared memory"])

    np.testing.assert_array_equal(first, second)


def test_repeated_call_same_provider_is_exactly_deterministic() -> None:
    provider = HashEmbeddingProvider(dim=64)
    texts = ["alpha beta", "gamma delta"]

    np.testing.assert_array_equal(provider.embed(texts), provider.embed(texts))


def test_different_providers_same_dim_are_exactly_deterministic() -> None:
    first = HashEmbeddingProvider(dim=64)
    second = HashEmbeddingProvider(dim=64)
    texts = ["alpha beta", "gamma delta"]

    np.testing.assert_array_equal(first.embed(texts), second.embed(texts))


def test_case_normalized_equivalent_text_is_same() -> None:
    provider = HashEmbeddingProvider()

    np.testing.assert_array_equal(
        provider.embed(["Linux MEMORY"]),
        provider.embed(["linux memory"]),
    )


def test_nfkc_equivalent_text_is_same() -> None:
    provider = HashEmbeddingProvider()
    first = "Ａｇｅｎｔ１２３"
    second = unicodedata.normalize("NFKC", first)

    np.testing.assert_array_equal(provider.embed([first]), provider.embed([second]))


def test_output_shape_dtype_layout_and_finiteness() -> None:
    provider = HashEmbeddingProvider(dim=31)
    result = provider.embed(["a", "b", "c"])

    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 31)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert np.all(np.isfinite(result))


def test_tokenized_non_empty_rows_are_unit_normalized() -> None:
    provider = HashEmbeddingProvider()
    result = provider.embed(["one", "one two", "共享 内存 状态"])

    norms = np.linalg.norm(result, axis=1)

    np.testing.assert_allclose(norms, np.ones(3), atol=1e-6)


@pytest.mark.parametrize("text", ["", "   \t\n", "!?.,---"])
def test_zero_token_text_returns_zero_vector(text: str) -> None:
    provider = HashEmbeddingProvider(dim=16)
    vector = provider.embed([text])[0]

    np.testing.assert_array_equal(vector, np.zeros(16, dtype=np.float32))
    assert float(np.linalg.norm(vector)) == 0.0


def test_empty_batch_returns_expected_shape_and_dtype() -> None:
    provider = HashEmbeddingProvider(dim=23)
    result = provider.embed([])

    assert result.shape == (0, 23)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous


@pytest.mark.parametrize("texts", [("a",), "a", iter(["a"]), None])
def test_non_list_input_is_rejected(texts: object) -> None:
    provider = HashEmbeddingProvider()

    with pytest.raises(TypeError):
        provider.embed(texts)  # type: ignore[arg-type]


def test_non_string_item_is_rejected() -> None:
    provider = HashEmbeddingProvider()

    with pytest.raises(TypeError):
        provider.embed(["ok", 1])  # type: ignore[list-item]


def test_different_text_normally_produces_different_vector() -> None:
    provider = HashEmbeddingProvider()
    result = provider.embed(["alpha beta", "gamma delta"])

    assert not np.array_equal(result[0], result[1])


def test_related_lexical_text_has_higher_similarity_than_unrelated_text() -> None:
    provider = HashEmbeddingProvider()
    query, related, unrelated = provider.embed(
        [
            "linux shared memory state",
            "linux shared memory state transfer",
            "banana telescope orchestra",
        ]
    )

    related_similarity = float(np.dot(query, related))
    unrelated_similarity = float(np.dot(query, unrelated))

    assert related_similarity > unrelated_similarity
