from __future__ import annotations

import numpy as np

from agentipc.memory.models import MemoryRecord
from agentipc.memory.scoring import (
    keyword_overlap_score,
    semantic_scores,
    tag_overlap_score,
)
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.refs import MemoryRef
from agentipc.providers.base import EmbeddingProvider


class MemoryService:
    def __init__(
        self,
        store: SQLiteMemoryStore,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
    ) -> None:
        if not isinstance(store, SQLiteMemoryStore):
            raise TypeError("store must be a SQLiteMemoryStore")
        if not isinstance(embedding_provider, EmbeddingProvider):
            raise TypeError("embedding_provider must satisfy EmbeddingProvider")
        if not isinstance(vector_index, VectorIndex):
            raise TypeError("vector_index must be a VectorIndex")
        self.store = store
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index

        if self.embedding_provider.dim != self.vector_index.dim:
            raise ValueError(
                "embedding_provider and vector_index dimension mismatch: "
                f"{self.embedding_provider.dim} != {self.vector_index.dim}"
            )

        self.vector_index.rebuild(self.store.list_records())

    def write(self, record: MemoryRecord) -> MemoryRecord:
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")

        embedded = self.embedding_provider.embed([record.summary])
        if not isinstance(embedded, np.ndarray):
            raise TypeError("embedding_provider.embed() must return a numpy.ndarray")

        expected_shape = (1, self.vector_index.dim)
        if embedded.shape != expected_shape:
            raise ValueError(
                "embedding_provider.embed() must return shape "
                f"{expected_shape}, got {embedded.shape}"
            )

        generated_vector = embedded[0]

        preflight_index = VectorIndex(self.vector_index.dim)
        preflight_index.add(record.memory_id, generated_vector)

        stored_record = record.model_copy(
            update={"embedding": generated_vector.tolist()},
        )
        self.store.write(stored_record)

        try:
            self.vector_index.add(stored_record.memory_id, generated_vector)
        except Exception:
            try:
                self.vector_index.rebuild(self.store.list_records())
            except Exception as rebuild_error:
                raise RuntimeError(
                    "memory was persisted, but vector index update/rebuild failed"
                ) from rebuild_error

        return stored_record

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self.store.get(memory_id)

    def retrieve(
        self,
        query: str,
        *,
        tags: list[str] | None = None,
        keywords: list[str] | None = None,
        top_k: int = 5,
    ) -> list[MemoryRef]:
        if not isinstance(query, str):
            raise TypeError("query must be a str")
        if tags is not None:
            if not isinstance(tags, list):
                raise TypeError("tags must be a list[str] or None")
            if not all(isinstance(value, str) for value in tags):
                raise TypeError("each tag must be a str")
        if keywords is not None:
            if not isinstance(keywords, list):
                raise TypeError("keywords must be a list[str] or None")
            if not all(isinstance(value, str) for value in keywords):
                raise TypeError("each keyword must be a str")
        if type(top_k) is not int:
            raise TypeError("top_k must be an int")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query_tags = [] if tags is None else tags
        query_keywords = [] if keywords is None else keywords

        records = self.store.list_records()
        semantic_map = semantic_scores(
            query,
            embedding_provider=self.embedding_provider,
            vector_index=self.vector_index,
        )

        scored: list[tuple[float, MemoryRecord]] = []
        for record in records:
            semantic = semantic_map.get(record.memory_id, 0.0)
            keyword = keyword_overlap_score(query_keywords, record.keywords)
            tag = tag_overlap_score(query_tags, record.tags)
            score = 0.60 * semantic + 0.25 * keyword + 0.15 * tag
            if score > 0.0:
                scored.append((score, record))

        scored.sort(key=lambda item: (-item[0], item[1].memory_id))

        return [
            MemoryRef(
                memory_id=record.memory_id,
                score=score,
                match_type="hybrid",
                summary=record.summary,
            )
            for score, record in scored[:top_k]
        ]

    def mark_used(
        self,
        memory_id: str,
        *,
        effective: bool | None = None,
    ) -> None:
        self.store.record_use(memory_id, effective=effective)
        return None
