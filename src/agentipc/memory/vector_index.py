from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from agentipc.memory.models import MemoryRecord


class VectorIndex:
    def __init__(self, dim: int) -> None:
        if type(dim) is not int:
            raise TypeError("dim must be an int")
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self._dim = dim
        self._vectors: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def __len__(self) -> int:
        return len(self._vectors)

    @staticmethod
    def _validate_memory_id(memory_id: object) -> str:
        if not isinstance(memory_id, str):
            raise TypeError("memory_id must be a str")
        if len(memory_id) == 0:
            raise ValueError("memory_id must be non-empty")
        return memory_id

    def _prepare_vector(self, vector: np.ndarray) -> np.ndarray:
        if not isinstance(vector, np.ndarray):
            raise TypeError("vector must be a numpy.ndarray")
        if vector.ndim != 1 or vector.shape != (self.dim,):
            raise ValueError(f"vector must have shape ({self.dim},)")

        dtype = vector.dtype
        if np.issubdtype(dtype, np.bool_):
            raise TypeError("vector dtype must be a real numeric dtype")
        if not np.issubdtype(dtype, np.number) or np.issubdtype(
            dtype, np.complexfloating
        ):
            raise TypeError("vector dtype must be a real numeric dtype")
        if not np.all(np.isfinite(vector)):
            raise ValueError("vector values must be finite")

        array64 = vector.astype(np.float64, copy=True)
        if not np.all(np.isfinite(array64)):
            raise ValueError("vector values must remain finite during conversion")

        max_abs = float(np.max(np.abs(array64))) if array64.size else 0.0
        if max_abs == 0.0:
            normalized64 = np.zeros(self.dim, dtype=np.float64)
        else:
            scaled = array64 / max_abs
            scaled_norm = float(np.linalg.norm(scaled))
            normalized64 = scaled / scaled_norm

        prepared = np.array(normalized64, dtype=np.float32, copy=True)
        if not np.all(np.isfinite(prepared)):
            raise ValueError("normalized vector must be finite")
        return prepared

    def add(self, memory_id: str, vector: np.ndarray) -> None:
        validated_id = self._validate_memory_id(memory_id)
        prepared = self._prepare_vector(vector)
        self._vectors[validated_id] = prepared

    def clear(self) -> None:
        self._vectors.clear()

    def search(
        self,
        query: np.ndarray,
        *,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        if type(top_k) is not int:
            raise TypeError("top_k must be an int")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        normalized_query = self._prepare_vector(query)
        if not self._vectors:
            return []

        results: list[tuple[str, float]] = []
        for memory_id, stored in self._vectors.items():
            score = float(np.dot(normalized_query, stored))
            score = float(np.clip(score, -1.0, 1.0))
            results.append((memory_id, score))

        results.sort(key=lambda item: (-item[1], item[0]))
        return results[:top_k]

    def rebuild(self, records: Iterable[MemoryRecord]) -> int:
        staged: dict[str, np.ndarray] = {}

        for record in records:
            if not isinstance(record, MemoryRecord):
                raise TypeError("records must contain only MemoryRecord values")

            if record.embedding is None:
                staged.pop(record.memory_id, None)
                continue

            vector = np.asarray(record.embedding, dtype=np.float32)
            prepared = self._prepare_vector(vector)
            staged[record.memory_id] = prepared

        self._vectors = staged
        return len(self._vectors)
