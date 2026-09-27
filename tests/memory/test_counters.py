import math

import pytest

from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.sqlite_store import SQLiteMemoryStore


def make_record(memory_id: str = "mem-1", **overrides) -> MemoryRecord:
    values = {
        "memory_id": memory_id,
        "source_agent": "planner",
        "created_at": 100.25,
        "task_topic": "topic",
        "summary": "summary",
        "memory_type": MemoryType.EVIDENCE,
        "tags": ["tag-b", "tag-a"],
        "keywords": ["Linux", "linux"],
        "embedding": [0.25, -0.5, 1.0],
        "payload": {"nested": {"ok": True}},
        "reuse_count": 4,
        "success_count": 2,
        "failure_count": 1,
        "last_accessed_at": 90.0,
    }
    values.update(overrides)
    return MemoryRecord(**values)


@pytest.mark.parametrize(
    ("effective", "success_delta", "failure_delta"),
    [(None, 0, 0), (True, 1, 0), (False, 0, 1)],
)
def test_record_use_updates_expected_counters(
    tmp_path, effective, success_delta: int, failure_delta: int
) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        updated = store.record_use(
            record.memory_id,
            effective=effective,
            accessed_at=123.5,
        )

    assert updated.reuse_count == record.reuse_count + 1
    assert updated.success_count == record.success_count + success_delta
    assert updated.failure_count == record.failure_count + failure_delta
    assert updated.last_accessed_at == 123.5


def test_record_use_accumulates_across_multiple_calls(tmp_path) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        store.record_use(record.memory_id, effective=True, accessed_at=101.0)
        store.record_use(record.memory_id, effective=False, accessed_at=102.0)
        updated = store.record_use(record.memory_id, effective=None, accessed_at=103.0)

    assert updated.reuse_count == 7
    assert updated.success_count == 3
    assert updated.failure_count == 2
    assert updated.last_accessed_at == 103.0


def test_record_use_preserves_non_counter_metadata(tmp_path) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        updated = store.record_use(record.memory_id, effective=True, accessed_at=222.0)

    before = record.model_dump(exclude={"reuse_count", "success_count", "failure_count", "last_accessed_at"})
    after = updated.model_dump(exclude={"reuse_count", "success_count", "failure_count", "last_accessed_at"})
    assert after == before


def test_record_use_returns_updated_memory_record(tmp_path) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        updated = store.record_use(record.memory_id, effective=True, accessed_at=222.0)
        loaded = store.get(record.memory_id)

    assert isinstance(updated, MemoryRecord)
    assert updated == loaded


def test_get_has_no_usage_side_effect(tmp_path) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        assert store.get(record.memory_id) == record
        assert store.get(record.memory_id) == record
        assert store.get(record.memory_id) == record


def test_accessed_at_none_uses_utc_timestamp(tmp_path, monkeypatch) -> None:
    record = make_record()
    monkeypatch.setattr("agentipc.memory.sqlite_store.utc_timestamp", lambda: 456.75)
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        updated = store.record_use(record.memory_id)
    assert updated.last_accessed_at == 456.75


def test_missing_memory_raises_key_error(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with pytest.raises(KeyError, match="missing"):
            store.record_use("missing", accessed_at=1.0)


@pytest.mark.parametrize("memory_id", [None, 1, b"mem-1"])
def test_record_use_rejects_non_string_id(tmp_path, memory_id) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with pytest.raises(TypeError):
            store.record_use(memory_id, accessed_at=1.0)


def test_record_use_rejects_empty_id(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with pytest.raises(ValueError):
            store.record_use("", accessed_at=1.0)


@pytest.mark.parametrize("effective", [1, 0, "true", [], object()])
def test_record_use_rejects_non_bool_effective(tmp_path, effective) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        with pytest.raises(TypeError):
            store.record_use(record.memory_id, effective=effective, accessed_at=1.0)


@pytest.mark.parametrize("accessed_at", [True, False, "1.0", object()])
def test_record_use_rejects_invalid_accessed_at_type(tmp_path, accessed_at) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        with pytest.raises(TypeError):
            store.record_use(record.memory_id, accessed_at=accessed_at)


@pytest.mark.parametrize("accessed_at", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_record_use_rejects_invalid_accessed_at_value(tmp_path, accessed_at) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        with pytest.raises(ValueError):
            store.record_use(record.memory_id, accessed_at=accessed_at)


def test_record_use_after_close_raises_runtime_error(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    store.close()
    with pytest.raises(RuntimeError):
        store.record_use("mem-1", accessed_at=1.0)


def test_record_use_persists_across_reopen(tmp_path) -> None:
    root = tmp_path / "memory"
    record = make_record()
    with SQLiteMemoryStore(root) as store:
        store.write(record)
        expected = store.record_use(record.memory_id, effective=False, accessed_at=321.0)

    with SQLiteMemoryStore(root) as reopened:
        assert reopened.get(record.memory_id) == expected


def test_atomic_increments_across_two_store_instances(tmp_path) -> None:
    root = tmp_path / "memory"
    record = make_record(reuse_count=8, success_count=5, failure_count=4)
    store_a = SQLiteMemoryStore(root)
    store_b = SQLiteMemoryStore(root)
    try:
        store_a.write(record)
        store_a.record_use(record.memory_id, effective=True, accessed_at=101.0)
        store_b.record_use(record.memory_id, effective=False, accessed_at=102.0)
        final = store_a.record_use(record.memory_id, effective=None, accessed_at=103.0)
    finally:
        store_a.close()
        store_b.close()

    assert final.reuse_count == 11
    assert final.success_count == 6
    assert final.failure_count == 5
    assert final.last_accessed_at == 103.0
    assert math.isfinite(final.last_accessed_at)
