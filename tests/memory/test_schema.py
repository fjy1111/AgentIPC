import sqlite3

import pytest

from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.sqlite_store import SQLiteMemoryStore


EXPECTED_COLUMNS = [
    "memory_id",
    "source_agent",
    "created_at",
    "task_topic",
    "summary",
    "memory_type",
    "tags_json",
    "keywords_json",
    "embedding_json",
    "payload_json",
    "reuse_count",
    "success_count",
    "failure_count",
    "last_accessed_at",
]


def make_record(memory_id: str = "mem-1") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        source_agent="planner",
        task_topic="topic",
        summary="summary",
        memory_type=MemoryType.EVIDENCE,
    )


def connect_readonly(store: SQLiteMemoryStore) -> sqlite3.Connection:
    connection = sqlite3.connect(store.db_path)
    connection.row_factory = sqlite3.Row
    return connection


def test_root_and_database_are_created(tmp_path) -> None:
    root = tmp_path / "memory"
    assert not root.exists()

    with SQLiteMemoryStore(root) as store:
        assert store.root == root
        assert store.db_path == root / "memory.sqlite3"
        assert root.is_dir()
        assert store.db_path.is_file()


def test_nested_root_is_created(tmp_path) -> None:
    root = tmp_path / "nested" / "deeper" / "memory"
    with SQLiteMemoryStore(root) as store:
        assert store.root.is_dir()
        assert store.db_path.is_file()


def test_file_root_is_rejected(tmp_path) -> None:
    root = tmp_path / "not-a-directory"
    root.write_text("file", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        SQLiteMemoryStore(root)


def test_memories_table_has_expected_schema(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with connect_readonly(store) as connection:
            rows = connection.execute("PRAGMA table_info(memories)").fetchall()

    assert [row["name"] for row in rows] == EXPECTED_COLUMNS
    by_name = {row["name"]: row for row in rows}
    assert by_name["memory_id"]["pk"] == 1

    required_not_null = {
        "source_agent",
        "created_at",
        "task_topic",
        "summary",
        "memory_type",
        "tags_json",
        "keywords_json",
        "payload_json",
        "reuse_count",
        "success_count",
        "failure_count",
    }
    for name in required_not_null:
        assert by_name[name]["notnull"] == 1

    assert by_name["embedding_json"]["notnull"] == 0
    assert by_name["last_accessed_at"]["notnull"] == 0


def test_required_indexes_exist(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with connect_readonly(store) as connection:
            rows = connection.execute("PRAGMA index_list(memories)").fetchall()

    index_names = {row["name"] for row in rows}
    assert "idx_memories_task_topic" in index_names
    assert "idx_memories_memory_type" in index_names
    assert "idx_memories_created_at" in index_names


def test_initialization_does_not_insert_business_rows(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        with connect_readonly(store) as connection:
            count = connection.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert count == 0


def test_same_root_can_be_initialized_twice(tmp_path) -> None:
    root = tmp_path / "memory"
    SQLiteMemoryStore(root).close()
    SQLiteMemoryStore(root).close()


def test_reinitialization_does_not_delete_existing_data(tmp_path) -> None:
    root = tmp_path / "memory"
    record = make_record()
    with SQLiteMemoryStore(root) as first:
        first.write(record)

    with SQLiteMemoryStore(root) as second:
        assert second.get("mem-1") == record


def test_close_is_idempotent(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    store.close()
    store.close()


def test_context_manager_closes_store(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        assert store.get("missing") is None

    with pytest.raises(RuntimeError):
        store.get("missing")


def test_context_manager_does_not_swallow_exception(tmp_path) -> None:
    class MarkerError(Exception):
        pass

    with pytest.raises(MarkerError):
        with SQLiteMemoryStore(tmp_path / "memory"):
            raise MarkerError("boom")
