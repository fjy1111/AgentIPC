import json
import sqlite3

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
        "payload": {"nested": {"ok": True}, "items": [1, 2, 3]},
        "reuse_count": 4,
        "success_count": 3,
        "failure_count": 1,
        "last_accessed_at": 125.5,
    }
    values.update(overrides)
    return MemoryRecord(**values)


def test_write_get_exact_round_trip(tmp_path) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        assert store.get(record.memory_id) == record


@pytest.mark.parametrize("memory_type", list(MemoryType))
def test_all_memory_types_round_trip(tmp_path, memory_type: MemoryType) -> None:
    record = make_record(memory_id=f"mem-{memory_type.value}", memory_type=memory_type)
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        assert store.get(record.memory_id) == record


def test_unicode_round_trip(tmp_path) -> None:
    record = make_record(
        task_topic="共享记忆 🚀",
        summary="中文总结 ✅",
        tags=["中文", "🧠"],
        keywords=["证据", "复用"],
        payload={"嵌套": {"结论": "保留 emoji 🎯"}},
    )
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        assert store.get(record.memory_id) == record


def test_tag_and_keyword_order_is_preserved(tmp_path) -> None:
    record = make_record(tags=["z", "a", "z"], keywords=["B", "a", "B"])
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        loaded = store.get(record.memory_id)
    assert loaded is not None
    assert loaded.tags == ["z", "a", "z"]
    assert loaded.keywords == ["B", "a", "B"]


def test_embedding_list_round_trip(tmp_path) -> None:
    record = make_record(embedding=[0.0, 1.25, -3.5])
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        assert store.get(record.memory_id) == record


def test_none_embedding_round_trip_and_is_sql_null(tmp_path) -> None:
    record = make_record(embedding=None)
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        assert store.get(record.memory_id) == record
        with sqlite3.connect(store.db_path) as connection:
            stored = connection.execute(
                "SELECT embedding_json FROM memories WHERE memory_id = ?",
                (record.memory_id,),
            ).fetchone()[0]
    assert stored is None


def test_nonzero_counters_and_last_accessed_at_round_trip(tmp_path) -> None:
    record = make_record(
        reuse_count=9,
        success_count=6,
        failure_count=2,
        last_accessed_at=999.75,
    )
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        assert store.get(record.memory_id) == record


def test_write_returns_the_supplied_record(tmp_path) -> None:
    record = make_record()
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        returned = store.write(record)
    assert returned is record


def test_missing_id_returns_none(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        assert store.get("missing") is None


@pytest.mark.parametrize("memory_id", [None, 1, b"mem-1"])
def test_get_non_string_id_raises_type_error(tmp_path, memory_id) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with pytest.raises(TypeError):
            store.get(memory_id)


def test_get_empty_id_raises_value_error(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with pytest.raises(ValueError):
            store.get("")


def test_get_does_not_strip_id(tmp_path) -> None:
    record = make_record(memory_id=" id ")
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        assert store.get(" id ") == record
        assert store.get("id") is None


def test_write_non_record_raises_type_error(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with pytest.raises(TypeError):
            store.write({"memory_id": "mem-1"})


def test_duplicate_id_raises_integrity_error_and_preserves_first_row(tmp_path) -> None:
    first = make_record(summary="first")
    duplicate = make_record(summary="second")

    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(first)
        with pytest.raises(sqlite3.IntegrityError):
            store.write(duplicate)
        assert store.get(first.memory_id) == first


def test_write_close_reopen_get_persists_record(tmp_path) -> None:
    root = tmp_path / "memory"
    record = make_record()

    store = SQLiteMemoryStore(root)
    store.write(record)
    store.close()

    with SQLiteMemoryStore(root) as reopened:
        assert reopened.get(record.memory_id) == record


def test_multiple_records_are_independent(tmp_path) -> None:
    first = make_record(memory_id="mem-1", summary="first")
    second = make_record(
        memory_id="mem-2",
        summary="second",
        memory_type=MemoryType.RESULT,
        embedding=None,
    )

    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(first)
        store.write(second)
        assert store.get("mem-1") == first
        assert store.get("mem-2") == second


def test_unsupported_payload_fails_before_insert(tmp_path) -> None:
    record = make_record(payload={"bad": object()})
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with pytest.raises(TypeError):
            store.write(record)
        assert store.get(record.memory_id) is None


def test_payload_nan_fails_before_insert(tmp_path) -> None:
    record = make_record(payload={"bad": float("nan")})
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with pytest.raises(ValueError):
            store.write(record)
        assert store.get(record.memory_id) is None


def test_json_columns_use_deterministic_compact_unicode_encoding(tmp_path) -> None:
    record = make_record(
        tags=["中文", "b"],
        keywords=["z", "a"],
        payload={"z": 1, "a": "中文"},
        embedding=[1.0, 2.0],
    )
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        store.write(record)
        with sqlite3.connect(store.db_path) as connection:
            row = connection.execute(
                "SELECT tags_json, keywords_json, embedding_json, payload_json "
                "FROM memories WHERE memory_id = ?",
                (record.memory_id,),
            ).fetchone()

    expected = lambda value: json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert row[0] == expected(record.tags)
    assert row[1] == expected(record.keywords)
    assert row[2] == expected(record.embedding)
    assert row[3] == expected(record.payload)
    assert "中文" in row[0]
    assert "中文" in row[3]


def test_operations_after_close_raise_runtime_error(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    record = make_record()
    store.close()

    with pytest.raises(RuntimeError):
        store.write(record)
    with pytest.raises(RuntimeError):
        store.get(record.memory_id)
