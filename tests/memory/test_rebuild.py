import numpy as np
import pytest

from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex


def make_record(memory_id: str, *, created_at: float, embedding, **overrides) -> MemoryRecord:
    values = {
        "memory_id": memory_id,
        "source_agent": "summarizer",
        "created_at": created_at,
        "task_topic": "共享记忆",
        "summary": f"summary-{memory_id}",
        "memory_type": MemoryType.RESULT,
        "tags": ["中文", "tag"],
        "keywords": ["复用", "memory"],
        "embedding": embedding,
        "payload": {"证据": memory_id},
        "reuse_count": 2,
        "success_count": 1,
        "failure_count": 1,
        "last_accessed_at": 500.0,
    }
    values.update(overrides)
    return MemoryRecord(**values)


def test_list_records_empty_database(tmp_path) -> None:
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        assert store.list_records() == []


def test_list_records_round_trip_order_and_no_side_effects(tmp_path) -> None:
    records = [
        make_record("b", created_at=10.0, embedding=[0.0, 1.0, 0.0]),
        make_record("a", created_at=10.0, embedding=[1.0, 0.0, 0.0]),
        make_record("c", created_at=5.0, embedding=None),
    ]
    with SQLiteMemoryStore(tmp_path / "memory") as store:
        for record in records:
            store.write(record)
        listed = store.list_records()
        listed_again = store.list_records()

    assert [record.memory_id for record in listed] == ["c", "a", "b"]
    expected = {record.memory_id: record for record in records}
    assert all(record == expected[record.memory_id] for record in listed)
    assert listed_again == listed


def test_list_records_after_close_raises_runtime_error(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory")
    store.close()
    with pytest.raises(RuntimeError):
        store.list_records()


def test_reopen_list_records_rebuild_and_search(tmp_path) -> None:
    root = tmp_path / "memory"
    a = make_record("A", created_at=1.0, embedding=[1.0, 0.0, 0.0])
    b = make_record("B", created_at=2.0, embedding=[0.0, 1.0, 0.0])
    c = make_record("C", created_at=3.0, embedding=None)

    with SQLiteMemoryStore(root) as store_a:
        store_a.write(a)
        store_a.write(b)
        store_a.write(c)

    with SQLiteMemoryStore(root) as store_b:
        records = store_b.list_records()

    index = VectorIndex(3)
    rebuilt = index.rebuild(records)
    results = index.search(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=3)

    assert rebuilt == 2
    assert len(index) == 2
    assert [memory_id for memory_id, _ in results] == ["A", "B"]
    assert results[0][1] == pytest.approx(1.0)
    assert "C" not in {memory_id for memory_id, _ in results}


def test_rebuild_replaces_stale_state(tmp_path) -> None:
    root = tmp_path / "memory"
    with SQLiteMemoryStore(root) as store:
        store.write(make_record("A", created_at=1.0, embedding=[1.0, 0.0]))
        store.write(make_record("B", created_at=2.0, embedding=[0.0, 1.0]))
        records = store.list_records()

    index = VectorIndex(2)
    index.add("stale", np.array([1.0, 1.0]))
    assert index.rebuild(records) == 2
    assert len(index) == 2
    assert {memory_id for memory_id, _ in index.search(np.array([1.0, 1.0]), top_k=10)} == {"A", "B"}


def test_all_none_embeddings_clear_existing_index(tmp_path) -> None:
    root = tmp_path / "memory"
    with SQLiteMemoryStore(root) as store:
        store.write(make_record("A", created_at=1.0, embedding=None))
        store.write(make_record("B", created_at=2.0, embedding=None))
        records = store.list_records()

    index = VectorIndex(2)
    index.add("stale", np.array([1.0, 0.0]))
    assert index.rebuild(records) == 0
    assert len(index) == 0
    assert index.search(np.array([1.0, 0.0])) == []


def test_dimension_mismatch_is_atomic(tmp_path) -> None:
    records = [
        make_record("good", created_at=1.0, embedding=[1.0, 0.0]),
        make_record("bad", created_at=2.0, embedding=[1.0, 0.0, 0.0]),
    ]
    index = VectorIndex(2)
    index.add("existing", np.array([1.0, 0.0]))
    before = index.search(np.array([1.0, 0.0]), top_k=10)

    with pytest.raises(ValueError):
        index.rebuild(records)

    assert len(index) == 1
    assert index.search(np.array([1.0, 0.0]), top_k=10) == before


def test_rebuild_rejects_non_memory_record_without_changing_old_index() -> None:
    index = VectorIndex(2)
    index.add("existing", np.array([1.0, 0.0]))
    before = index.search(np.array([1.0, 0.0]))

    with pytest.raises(TypeError):
        index.rebuild([make_record("A", created_at=1.0, embedding=[1.0, 0.0]), {"memory_id": "bad"}])

    assert index.search(np.array([1.0, 0.0])) == before


def test_rebuild_accepts_generator() -> None:
    index = VectorIndex(2)
    records = (
        make_record(memory_id, created_at=float(i), embedding=embedding)
        for i, (memory_id, embedding) in enumerate(
            [("A", [1.0, 0.0]), ("B", [0.0, 1.0])], start=1
        )
    )
    assert index.rebuild(records) == 2


def test_rebuild_duplicate_id_last_record_wins() -> None:
    index = VectorIndex(2)
    first = make_record("same", created_at=1.0, embedding=[1.0, 0.0])
    second = make_record("same", created_at=2.0, embedding=[0.0, 1.0])
    assert index.rebuild([first, second]) == 1
    assert index.search(np.array([0.0, 1.0]), top_k=1)[0][1] == pytest.approx(1.0)


def test_rebuild_duplicate_id_last_none_removes_prior_embedding() -> None:
    index = VectorIndex(2)
    first = make_record("same", created_at=1.0, embedding=[1.0, 0.0])
    second = make_record("same", created_at=2.0, embedding=None)
    assert index.rebuild([first, second]) == 0
    assert len(index) == 0
