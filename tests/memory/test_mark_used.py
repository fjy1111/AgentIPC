import math

import pytest

from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.providers.hash_embedding import HashEmbeddingProvider


def make_record() -> MemoryRecord:
    return MemoryRecord(
        memory_id="mem-1",
        source_agent="summarizer",
        created_at=100.0,
        task_topic="topic",
        summary="summary",
        memory_type=MemoryType.RESULT,
        reuse_count=2,
        success_count=1,
        failure_count=1,
    )


def make_service(tmp_path) -> tuple[SQLiteMemoryStore, MemoryService]:
    store = SQLiteMemoryStore(tmp_path / "memory")
    provider = HashEmbeddingProvider(8)
    return store, MemoryService(store, provider, VectorIndex(provider.dim))


@pytest.mark.parametrize(
    ("effective", "success_delta", "failure_delta"),
    [(True, 1, 0), (False, 0, 1), (None, 0, 0)],
)
def test_mark_used_updates_expected_counters_and_returns_none(
    tmp_path, effective, success_delta: int, failure_delta: int
) -> None:
    store, service = make_service(tmp_path)
    record = make_record()
    try:
        service.write(record)
        result = service.mark_used(record.memory_id, effective=effective)
        updated = service.get(record.memory_id)
        assert result is None
        assert updated is not None
        assert updated.reuse_count == record.reuse_count + 1
        assert updated.success_count == record.success_count + success_delta
        assert updated.failure_count == record.failure_count + failure_delta
        assert updated.last_accessed_at is not None
        assert math.isfinite(updated.last_accessed_at)
        assert updated.last_accessed_at >= 0.0
    finally:
        store.close()


def test_mark_used_repeated_calls_accumulate(tmp_path) -> None:
    store, service = make_service(tmp_path)
    record = make_record()
    try:
        service.write(record)
        service.mark_used(record.memory_id, effective=True)
        service.mark_used(record.memory_id, effective=False)
        service.mark_used(record.memory_id, effective=None)
        updated = service.get(record.memory_id)
        assert updated is not None
        assert updated.reuse_count == 5
        assert updated.success_count == 2
        assert updated.failure_count == 2
        assert updated.last_accessed_at is not None
    finally:
        store.close()


def test_mark_used_missing_id_preserves_key_error(tmp_path) -> None:
    store, service = make_service(tmp_path)
    try:
        with pytest.raises(KeyError, match="missing"):
            service.mark_used("missing")
    finally:
        store.close()


@pytest.mark.parametrize("effective", [1, 0, "true", [], {}, object()])
def test_mark_used_invalid_effective_preserves_type_error(tmp_path, effective) -> None:
    store, service = make_service(tmp_path)
    record = make_record()
    try:
        service.write(record)
        with pytest.raises(TypeError):
            service.mark_used(record.memory_id, effective=effective)
    finally:
        store.close()
