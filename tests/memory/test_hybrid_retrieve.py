import math

import numpy as np
import pytest

from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.refs import MemoryRef


class MappingEmbeddingProvider:
    def __init__(self, mapping: dict[str, list[float]], *, dim: int = 2) -> None:
        self.mapping = mapping
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.array([self.mapping.get(text, [0.0] * self.dim) for text in texts], dtype=np.float32)


def make_record(
    memory_id: str,
    summary: str,
    *,
    task_topic="topic",
    memory_type=MemoryType.RESULT,
    keywords=None,
    tags=None,
    embedding=None,
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        source_agent="summarizer",
        created_at=100.0,
        task_topic=task_topic,
        summary=summary,
        memory_type=memory_type,
        keywords=[] if keywords is None else keywords,
        tags=[] if tags is None else tags,
        embedding=embedding,
        payload={},
    )


def build_service(tmp_path, mapping: dict[str, list[float]]) -> tuple[SQLiteMemoryStore, MemoryService]:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = MappingEmbeddingProvider(mapping)
    service = MemoryService(store, provider, VectorIndex(provider.dim))
    return store, service


def test_hybrid_weights_change_ranking_and_return_expected_refs(tmp_path) -> None:
    root = math.sqrt(3.0) / 2.0
    store, service = build_service(
        tmp_path,
        {"Q": [1.0, 0.0], "summary-A": [1.0, 0.0], "summary-B": [0.5, root]},
    )
    try:
        service.write(make_record("A", "summary-A"))
        service.write(make_record("B", "summary-B", keywords=["kw"], tags=["tag"]))

        refs = service.retrieve("Q", keywords=["kw"], tags=["tag"], top_k=2)

        assert [ref.memory_id for ref in refs] == ["B", "A"]
        assert all(isinstance(ref, MemoryRef) for ref in refs)
        assert refs[0].score == pytest.approx(0.70, abs=1e-6)
        assert refs[1].score == pytest.approx(0.60, abs=1e-6)
        assert refs[0].match_type == refs[1].match_type == "hybrid"
        assert refs[0].summary == "summary-B"
        assert refs[1].summary == "summary-A"
    finally:
        store.close()


def test_top_k_is_applied_after_full_hybrid_ranking(tmp_path) -> None:
    root = math.sqrt(3.0) / 2.0
    store, service = build_service(
        tmp_path,
        {"Q": [1.0, 0.0], "summary-A": [1.0, 0.0], "summary-B": [0.5, root]},
    )
    try:
        service.write(make_record("A", "summary-A"))
        service.write(make_record("B", "summary-B", keywords=["kw"], tags=["tag"]))
        refs = service.retrieve("Q", keywords=["kw"], tags=["tag"], top_k=1)
        assert [ref.memory_id for ref in refs] == ["B"]
    finally:
        store.close()


def test_tie_break_is_memory_id_ascending(tmp_path) -> None:
    store, service = build_service(
        tmp_path,
        {"Q": [1.0, 0.0], "summary-a": [0.0, 1.0], "summary-b": [0.0, 1.0]},
    )
    try:
        service.write(make_record("b", "summary-b", keywords=["kw"]))
        service.write(make_record("a", "summary-a", keywords=["kw"]))
        refs = service.retrieve("Q", keywords=["kw"], top_k=2)
        assert [ref.memory_id for ref in refs] == ["a", "b"]
        assert refs[0].score == pytest.approx(0.25)
        assert refs[1].score == pytest.approx(0.25)
    finally:
        store.close()


def test_zero_score_records_are_excluded_and_no_matches_returns_empty(tmp_path) -> None:
    store, service = build_service(tmp_path, {"Q": [1.0, 0.0], "unrelated": [0.0, 1.0]})
    try:
        service.write(make_record("zero", "unrelated"))
        assert service.retrieve("Q", top_k=5) == []
    finally:
        store.close()


def test_zero_score_exact_task_result_is_included(tmp_path) -> None:
    store, service = build_service(
        tmp_path,
        {"exact-query": [1.0, 0.0], "unrelated": [0.0, 1.0]},
    )
    try:
        service.write(
            make_record(
                "exact",
                "unrelated",
                task_topic="exact-query",
            )
        )

        refs = service.retrieve("exact-query")

        assert len(refs) == 1
        assert refs[0].memory_id == "exact"
        assert refs[0].score == pytest.approx(0.0)
        assert refs[0].match_type == "exact_task"
    finally:
        store.close()


def test_exact_task_result_has_priority_before_top_k(tmp_path) -> None:
    store, service = build_service(
        tmp_path,
        {
            "exact-query": [1.0, 0.0],
            "exact-unrelated": [0.0, 1.0],
            "high-hybrid": [1.0, 0.0],
        },
    )
    try:
        service.write(
            make_record(
                "A",
                "exact-unrelated",
                task_topic="exact-query",
            )
        )
        service.write(
            make_record(
                "B",
                "high-hybrid",
                task_topic="other-query",
                keywords=["kw"],
                tags=["tag"],
            )
        )

        refs = service.retrieve(
            "exact-query",
            keywords=["kw"],
            tags=["tag"],
            top_k=1,
        )

        assert [ref.memory_id for ref in refs] == ["A"]
        assert refs[0].score == pytest.approx(0.0)
        assert refs[0].match_type == "exact_task"
    finally:
        store.close()


def test_zero_score_exact_task_non_result_is_not_included(tmp_path) -> None:
    store, service = build_service(
        tmp_path,
        {"exact-query": [1.0, 0.0], "unrelated": [0.0, 1.0]},
    )
    try:
        service.write(
            make_record(
                "evidence",
                "unrelated",
                task_topic="exact-query",
                memory_type=MemoryType.EVIDENCE,
            )
        )

        assert service.retrieve("exact-query") == []
    finally:
        store.close()


def test_none_tags_and_keywords_are_equivalent_to_empty_lists(tmp_path) -> None:
    store, service = build_service(tmp_path, {"Q": [1.0, 0.0], "summary": [1.0, 0.0]})
    try:
        service.write(make_record("mem", "summary"))
        none_result = service.retrieve("Q", tags=None, keywords=None)
        empty_result = service.retrieve("Q", tags=[], keywords=[])
        assert none_result == empty_result
    finally:
        store.close()


def test_sqlite_record_without_embedding_can_match_keyword_and_tag(tmp_path) -> None:
    store, service = build_service(tmp_path, {"Q": [1.0, 0.0]})
    legacy = make_record("legacy", "legacy summary", keywords=["kw"], tags=["tag"], embedding=None)
    try:
        store.write(legacy)
        refs = service.retrieve("Q", keywords=["kw"], tags=["tag"], top_k=5)
        assert [ref.memory_id for ref in refs] == ["legacy"]
        assert refs[0].score == pytest.approx(0.40)
    finally:
        store.close()


def test_retrieve_has_no_usage_side_effects(tmp_path) -> None:
    store, service = build_service(tmp_path, {"Q": [1.0, 0.0], "summary": [1.0, 0.0]})
    try:
        stored = service.write(make_record("mem", "summary"))
        before = service.get(stored.memory_id)
        service.retrieve("Q")
        service.retrieve("Q")
        after = service.get(stored.memory_id)
        assert after == before
    finally:
        store.close()


@pytest.mark.parametrize("top_k", [True, False, 1.0, "1", None])
def test_invalid_top_k_type_raises_type_error(tmp_path, top_k) -> None:
    store, service = build_service(tmp_path, {})
    try:
        with pytest.raises(TypeError):
            service.retrieve("Q", top_k=top_k)
    finally:
        store.close()


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_raises_value_error(tmp_path, top_k) -> None:
    store, service = build_service(tmp_path, {})
    try:
        with pytest.raises(ValueError):
            service.retrieve("Q", top_k=top_k)
    finally:
        store.close()


@pytest.mark.parametrize("query", [None, 1, b"Q", ["Q"]])
def test_invalid_query_type_raises_type_error(tmp_path, query) -> None:
    store, service = build_service(tmp_path, {})
    try:
        with pytest.raises(TypeError):
            service.retrieve(query)
    finally:
        store.close()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tags": "tag"},
        {"tags": ("tag",)},
        {"tags": ["tag", 1]},
        {"keywords": "kw"},
        {"keywords": ("kw",)},
        {"keywords": ["kw", None]},
    ],
)
def test_invalid_tag_or_keyword_input_raises_type_error(tmp_path, kwargs) -> None:
    store, service = build_service(tmp_path, {})
    try:
        with pytest.raises(TypeError):
            service.retrieve("Q", **kwargs)
    finally:
        store.close()
