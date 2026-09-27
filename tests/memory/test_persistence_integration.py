import math

import pytest

from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.refs import MemoryRef
from agentipc.providers.hash_embedding import HashEmbeddingProvider


EMBEDDING_DIM = 64
MEMORY_ID = "mem-task-a-network-recovery"
TASK_TOPIC = "openEuler network recovery"
SUMMARY = "restart NetworkManager restores network connectivity"
TAGS = ["openeuler", "network"]
KEYWORDS = ["networkmanager", "connectivity"]
ACTION = "systemctl restart NetworkManager"
PAYLOAD = {"action": ACTION, "result": "connectivity restored"}


def test_task_a_write_close_reopen_task_b_retrieve_and_mark_used(tmp_path) -> None:
    root = tmp_path / "memory"

    store_a = SQLiteMemoryStore(root)
    provider_a = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    index_a = VectorIndex(provider_a.dim)
    service_a = MemoryService(store_a, provider_a, index_a)

    record_a = MemoryRecord(
        memory_id=MEMORY_ID,
        source_agent="summarizer",
        created_at=100.0,
        task_topic=TASK_TOPIC,
        summary=SUMMARY,
        memory_type=MemoryType.EXPERIENCE,
        tags=TAGS,
        keywords=KEYWORDS,
        embedding=None,
        payload=PAYLOAD,
    )

    try:
        assert store_a.list_records() == []

        stored_a = service_a.write(record_a)

        assert stored_a.embedding is not None
        assert len(stored_a.embedding) == EMBEDDING_DIM
        assert service_a.get(MEMORY_ID) == stored_a
        assert len(index_a) == 1
        assert stored_a.reuse_count == 0
        assert stored_a.success_count == 0
        assert stored_a.failure_count == 0
        assert stored_a.last_accessed_at is None
    finally:
        store_a.close()

    del service_a
    del store_a
    del index_a
    del provider_a
    del stored_a
    del record_a

    store_b = SQLiteMemoryStore(root)
    provider_b = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    index_b = VectorIndex(provider_b.dim)
    assert len(index_b) == 0

    service_b = MemoryService(store_b, provider_b, index_b)
    try:
        assert len(index_b) == 1

        persisted = service_b.get(MEMORY_ID)
        assert persisted is not None
        assert persisted.memory_id == MEMORY_ID
        assert persisted.summary == SUMMARY
        assert persisted.task_topic == TASK_TOPIC
        assert persisted.memory_type == MemoryType.EXPERIENCE
        assert persisted.tags == TAGS
        assert persisted.keywords == KEYWORDS
        assert persisted.payload == PAYLOAD
        assert persisted.embedding is not None
        assert persisted.embedding == pytest.approx(
            provider_b.embed([SUMMARY])[0].tolist()
        )

        refs = service_b.retrieve(
            persisted.summary,
            top_k=5,
        )

        assert len(refs) >= 1
        assert isinstance(refs[0], MemoryRef)
        assert refs[0].memory_id == MEMORY_ID
        assert refs[0].match_type == "hybrid"
        assert refs[0].summary == persisted.summary
        assert refs[0].score == pytest.approx(0.60, abs=1e-6)

        resolved = service_b.get(refs[0].memory_id)
        assert resolved is not None
        task_b_selected_action = resolved.payload["action"]
        assert task_b_selected_action == ACTION

        before_use = service_b.get(MEMORY_ID)
        assert before_use is not None
        assert before_use.reuse_count == 0
        assert before_use.success_count == 0
        assert before_use.failure_count == 0
        assert before_use.last_accessed_at is None

        result = service_b.mark_used(
            refs[0].memory_id,
            effective=True,
        )
        assert result is None

        after_use = service_b.get(MEMORY_ID)
        assert after_use is not None
        assert after_use.reuse_count == 1
        assert after_use.success_count == 1
        assert after_use.failure_count == 0
        assert after_use.last_accessed_at is not None
        assert math.isfinite(after_use.last_accessed_at)
        assert after_use.last_accessed_at >= 0.0
    finally:
        store_b.close()

    store_c = SQLiteMemoryStore(root)
    try:
        persisted_after_use = store_c.get(MEMORY_ID)
        assert persisted_after_use is not None
        assert persisted_after_use.reuse_count == 1
        assert persisted_after_use.success_count == 1
        assert persisted_after_use.failure_count == 0
        assert persisted_after_use.last_accessed_at is not None
        assert math.isfinite(persisted_after_use.last_accessed_at)
        assert persisted_after_use.last_accessed_at >= 0.0
    finally:
        store_c.close()
