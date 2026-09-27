import sqlite3

import numpy as np
import pytest

from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.providers.hash_embedding import HashEmbeddingProvider


def make_record(memory_id: str = "mem-1", **overrides) -> MemoryRecord:
    values = {
        "memory_id": memory_id,
        "source_agent": "summarizer",
        "created_at": 100.0,
        "task_topic": "memory",
        "summary": "linux shared memory",
        "memory_type": MemoryType.RESULT,
        "tags": ["linux"],
        "keywords": ["memory"],
        "payload": {"ok": True},
    }
    values.update(overrides)
    return MemoryRecord(**values)


class RecordingProvider:
    def __init__(self, vector: np.ndarray) -> None:
        self._vector = vector
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return int(self._vector.shape[1]) if self._vector.ndim == 2 else 2

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts)
        return self._vector.copy()


class RaisingProvider:
    @property
    def dim(self) -> int:
        return 2

    def embed(self, texts: list[str]) -> np.ndarray:
        raise LookupError("embedding failed")


class FailingAddOnceIndex(VectorIndex):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.fail_next_add = True
        self.rebuild_calls = 0

    def add(self, memory_id: str, vector: np.ndarray) -> None:
        if self.fail_next_add:
            self.fail_next_add = False
            raise RuntimeError("injected add failure")
        super().add(memory_id, vector)

    def rebuild(self, records) -> int:
        self.rebuild_calls += 1
        return super().rebuild(records)


class FailingAddAndRebuildIndex(VectorIndex):
    def __init__(self, dim: int) -> None:
        super().__init__(dim)
        self.initialized = False

    def rebuild(self, records) -> int:
        if not self.initialized:
            self.initialized = True
            return super().rebuild(records)
        raise RuntimeError("injected rebuild failure")

    def add(self, memory_id: str, vector: np.ndarray) -> None:
        raise RuntimeError("injected add failure")


def test_write_happy_path_db_and_index_visible(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = HashEmbeddingProvider(dim=32)
    index = VectorIndex(provider.dim)
    service = MemoryService(store, provider, index)
    try:
        stored = service.write(make_record())
        loaded = store.get("mem-1")
        assert isinstance(stored, MemoryRecord)
        assert loaded == stored
        assert stored.embedding is not None
        assert len(stored.embedding) == provider.dim
        query_vector = provider.embed([stored.summary])[0]
        assert index.search(query_vector, top_k=1)[0][0] == "mem-1"
    finally:
        store.close()


def test_existing_caller_embedding_is_overwritten_without_mutating_caller(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = HashEmbeddingProvider(dim=8)
    index = VectorIndex(provider.dim)
    service = MemoryService(store, provider, index)
    caller = make_record(embedding=[99.0])
    before = caller.model_copy(deep=True)
    try:
        stored = service.write(caller)
        expected = provider.embed([caller.summary])[0].tolist()
        assert stored.embedding == pytest.approx(expected)
        assert caller == before
        assert caller.embedding == [99.0]
    finally:
        store.close()


def test_write_uses_summary_as_only_embedding_text(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = RecordingProvider(np.array([[1.0, 0.0]], dtype=np.float32))
    service = MemoryService(store, provider, VectorIndex(2))
    record = make_record(summary="summary only", task_topic="do not include", tags=["tag"], keywords=["keyword"])
    try:
        service.write(record)
        assert provider.calls == [["summary only"]]
    finally:
        store.close()


def test_duplicate_db_failure_does_not_mutate_real_index(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = RecordingProvider(np.array([[1.0, 0.0]], dtype=np.float32))
    index = VectorIndex(2)
    service = MemoryService(store, provider, index)
    try:
        first = service.write(make_record(summary="first"))
        before = index.search(np.array([1.0, 0.0]), top_k=5)
        provider._vector = np.array([[0.0, 1.0]], dtype=np.float32)
        with pytest.raises(sqlite3.IntegrityError):
            service.write(make_record(summary="second"))
        assert store.get("mem-1") == first
        assert index.search(np.array([1.0, 0.0]), top_k=5) == before
    finally:
        store.close()


def test_embedding_failure_occurs_before_db_commit_and_index_change(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    index = VectorIndex(2)
    service = MemoryService(store, RaisingProvider(), index)
    try:
        with pytest.raises(LookupError, match="embedding failed"):
            service.write(make_record())
        assert store.get("mem-1") is None
        assert len(index) == 0
    finally:
        store.close()


@pytest.mark.parametrize(
    "output, expected_error",
    [
        (np.array([1.0, 0.0], dtype=np.float32), ValueError),
        (np.array([[np.nan, 0.0]], dtype=np.float32), ValueError),
    ],
)
def test_invalid_provider_output_fails_before_db_commit(tmp_path, output, expected_error) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = RecordingProvider(output)
    index = VectorIndex(2)
    service = MemoryService(store, provider, index)
    try:
        with pytest.raises(expected_error):
            service.write(make_record())
        assert store.get("mem-1") is None
        assert len(index) == 0
    finally:
        store.close()


def test_post_commit_index_add_failure_recovers_by_rebuild(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = RecordingProvider(np.array([[1.0, 0.0]], dtype=np.float32))
    index = FailingAddOnceIndex(2)
    service = MemoryService(store, provider, index)
    constructor_rebuild_calls = index.rebuild_calls
    try:
        stored = service.write(make_record())
        assert store.get("mem-1") == stored
        assert index.rebuild_calls == constructor_rebuild_calls + 1
        assert index.search(np.array([1.0, 0.0]), top_k=1)[0][0] == "mem-1"
    finally:
        store.close()


def test_post_commit_add_and_rebuild_failure_raises_runtime_error_but_keeps_db(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = RecordingProvider(np.array([[1.0, 0.0]], dtype=np.float32))
    index = FailingAddAndRebuildIndex(2)
    service = MemoryService(store, provider, index)
    try:
        with pytest.raises(RuntimeError, match="memory was persisted") as exc_info:
            service.write(make_record())
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert store.get("mem-1") is not None
    finally:
        store.close()


def test_constructor_rebuilds_index_from_sqlite(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    record = make_record(embedding=[1.0, 0.0])
    store.write(record)
    index = VectorIndex(2)
    try:
        MemoryService(store, RecordingProvider(np.array([[1.0, 0.0]], dtype=np.float32)), index)
        assert index.search(np.array([1.0, 0.0]), top_k=1)[0][0] == record.memory_id
    finally:
        store.close()


def test_constructor_dimension_mismatch_raises(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    try:
        with pytest.raises(ValueError, match="dimension mismatch"):
            MemoryService(store, HashEmbeddingProvider(dim=3), VectorIndex(2))
    finally:
        store.close()


@pytest.mark.parametrize(
    "store, provider, index",
    [
        (object(), HashEmbeddingProvider(2), VectorIndex(2)),
        (None, object(), VectorIndex(2)),
        (None, HashEmbeddingProvider(2), object()),
    ],
)
def test_constructor_rejects_invalid_dependencies(tmp_path, store, provider, index) -> None:
    real_store = SQLiteMemoryStore(tmp_path / "memory")
    if store is None:
        store = real_store
    try:
        with pytest.raises(TypeError):
            MemoryService(store, provider, index)
    finally:
        real_store.close()


def test_get_is_pure_read(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = HashEmbeddingProvider(8)
    service = MemoryService(store, provider, VectorIndex(8))
    try:
        stored = service.write(make_record())
        first = service.get(stored.memory_id)
        second = service.get(stored.memory_id)
        assert first == stored
        assert second == stored
        assert second.reuse_count == 0
        assert second.success_count == 0
        assert second.failure_count == 0
        assert second.last_accessed_at is None
    finally:
        store.close()
