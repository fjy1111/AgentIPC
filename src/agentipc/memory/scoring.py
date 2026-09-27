import unicodedata

import numpy as np

from agentipc.memory.vector_index import VectorIndex
from agentipc.providers.base import EmbeddingProvider


def _normalize_term(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def keyword_overlap_score(
    query_keywords: list[str],
    record_keywords: list[str],
) -> float:
    if not isinstance(query_keywords, list):
        raise TypeError("query_keywords must be a list[str]")
    if not isinstance(record_keywords, list):
        raise TypeError("record_keywords must be a list[str]")
    if not all(isinstance(value, str) for value in query_keywords):
        raise TypeError("each query keyword must be a str")
    if not all(isinstance(value, str) for value in record_keywords):
        raise TypeError("each record keyword must be a str")

    query_terms = {
        normalized
        for value in query_keywords
        if (normalized := _normalize_term(value))
    }
    record_terms = {
        normalized
        for value in record_keywords
        if (normalized := _normalize_term(value))
    }

    if not query_terms:
        return 0.0
    return float(len(query_terms & record_terms) / len(query_terms))


def tag_overlap_score(
    query_tags: list[str],
    record_tags: list[str],
) -> float:
    if not isinstance(query_tags, list):
        raise TypeError("query_tags must be a list[str]")
    if not isinstance(record_tags, list):
        raise TypeError("record_tags must be a list[str]")
    if not all(isinstance(value, str) for value in query_tags):
        raise TypeError("each query tag must be a str")
    if not all(isinstance(value, str) for value in record_tags):
        raise TypeError("each record tag must be a str")

    query_terms = {
        normalized
        for value in query_tags
        if (normalized := _normalize_term(value))
    }
    record_terms = {
        normalized
        for value in record_tags
        if (normalized := _normalize_term(value))
    }

    if not query_terms:
        return 0.0
    return float(len(query_terms & record_terms) / len(query_terms))


def semantic_scores(
    query: str,
    *,
    embedding_provider: EmbeddingProvider,
    vector_index: VectorIndex,
) -> dict[str, float]:
    if not isinstance(query, str):
        raise TypeError("query must be a str")
    if not isinstance(embedding_provider, EmbeddingProvider):
        raise TypeError("embedding_provider must satisfy EmbeddingProvider")
    if not isinstance(vector_index, VectorIndex):
        raise TypeError("vector_index must be a VectorIndex")
    provider_dim = embedding_provider.dim
    if provider_dim != vector_index.dim:
        raise ValueError(
            "embedding_provider and vector_index dimension mismatch: "
            f"{provider_dim} != {vector_index.dim}"
        )

    if len(vector_index) == 0:
        return {}

    embedded = embedding_provider.embed([query])
    if not isinstance(embedded, np.ndarray):
        raise TypeError("embedding_provider.embed() must return a numpy.ndarray")

    expected_shape = (1, vector_index.dim)
    if embedded.shape != expected_shape:
        raise ValueError(
            "embedding_provider.embed() must return shape "
            f"{expected_shape}, got {embedded.shape}"
        )

    query_vector = embedded[0]
    raw_results = vector_index.search(
        query_vector,
        top_k=max(1, len(vector_index)),
    )

    scores: dict[str, float] = {}
    for memory_id, cosine in raw_results:
        semantic_score = max(0.0, cosine)
        semantic_score = min(1.0, semantic_score)
        scores[memory_id] = float(semantic_score)
    return scores
