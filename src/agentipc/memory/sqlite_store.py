from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from agentipc.memory.models import MemoryRecord


_DB_FILENAME = "memory.sqlite3"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    source_agent TEXT NOT NULL,
    created_at REAL NOT NULL,
    task_topic TEXT NOT NULL,
    summary TEXT NOT NULL,

    memory_type TEXT NOT NULL
        CHECK (memory_type IN ('evidence', 'experience', 'result')),

    tags_json TEXT NOT NULL,
    keywords_json TEXT NOT NULL,
    embedding_json TEXT,
    payload_json TEXT NOT NULL,

    reuse_count INTEGER NOT NULL
        CHECK (reuse_count >= 0),

    success_count INTEGER NOT NULL
        CHECK (success_count >= 0),

    failure_count INTEGER NOT NULL
        CHECK (failure_count >= 0),

    last_accessed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_memories_task_topic
ON memories(task_topic);

CREATE INDEX IF NOT EXISTS idx_memories_memory_type
ON memories(memory_type);

CREATE INDEX IF NOT EXISTS idx_memories_created_at
ON memories(created_at);
"""

_INSERT_SQL = """
INSERT INTO memories (
    memory_id,
    source_agent,
    created_at,
    task_topic,
    summary,
    memory_type,
    tags_json,
    keywords_json,
    embedding_json,
    payload_json,
    reuse_count,
    success_count,
    failure_count,
    last_accessed_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_SELECT_BY_ID_SQL = """
SELECT
    memory_id,
    source_agent,
    created_at,
    task_topic,
    summary,
    memory_type,
    tags_json,
    keywords_json,
    embedding_json,
    payload_json,
    reuse_count,
    success_count,
    failure_count,
    last_accessed_at
FROM memories
WHERE memory_id = ?
"""


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


class SQLiteMemoryStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise NotADirectoryError(f"memory root is not a directory: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)

        self.db_path = self.root / _DB_FILENAME
        self._connection: sqlite3.Connection | None = sqlite3.connect(self.db_path)
        self._connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _ensure_open(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("SQLiteMemoryStore is closed")
        return self._connection

    def _initialize_schema(self) -> None:
        connection = self._ensure_open()
        with connection:
            connection.executescript(_SCHEMA)

    def write(self, record: MemoryRecord) -> MemoryRecord:
        connection = self._ensure_open()
        if not isinstance(record, MemoryRecord):
            raise TypeError("record must be a MemoryRecord")

        tags_json = _json_dumps(record.tags)
        keywords_json = _json_dumps(record.keywords)
        embedding_json = (
            None if record.embedding is None else _json_dumps(record.embedding)
        )
        payload_json = _json_dumps(record.payload)

        values = (
            record.memory_id,
            record.source_agent,
            record.created_at,
            record.task_topic,
            record.summary,
            record.memory_type.value,
            tags_json,
            keywords_json,
            embedding_json,
            payload_json,
            record.reuse_count,
            record.success_count,
            record.failure_count,
            record.last_accessed_at,
        )

        with connection:
            connection.execute(_INSERT_SQL, values)

        return record

    def get(self, memory_id: str) -> MemoryRecord | None:
        connection = self._ensure_open()
        if not isinstance(memory_id, str):
            raise TypeError("memory_id must be a str")
        if len(memory_id) == 0:
            raise ValueError("memory_id must be non-empty")

        row = connection.execute(_SELECT_BY_ID_SQL, (memory_id,)).fetchone()
        if row is None:
            return None

        return MemoryRecord(
            memory_id=row["memory_id"],
            source_agent=row["source_agent"],
            created_at=row["created_at"],
            task_topic=row["task_topic"],
            summary=row["summary"],
            memory_type=row["memory_type"],
            tags=json.loads(row["tags_json"]),
            keywords=json.loads(row["keywords_json"]),
            embedding=(
                None
                if row["embedding_json"] is None
                else json.loads(row["embedding_json"])
            ),
            payload=json.loads(row["payload_json"]),
            reuse_count=row["reuse_count"],
            success_count=row["success_count"],
            failure_count=row["failure_count"],
            last_accessed_at=row["last_accessed_at"],
        )

    def close(self) -> None:
        if self._connection is None:
            return
        self._connection.close()
        self._connection = None

    def __enter__(self) -> SQLiteMemoryStore:
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
