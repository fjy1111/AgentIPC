from __future__ import annotations

import numpy as np

from agentipc.artifacts.store import ArtifactStore
from agentipc.memory.models import MemoryRecord
from agentipc.memory.service import MemoryService
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef
from agentipc.state.hub import StateHub


class ReferenceResolver:
    def __init__(
        self,
        *,
        state_hub: StateHub,
        artifact_store: ArtifactStore,
        memory_service: MemoryService,
    ) -> None:
        if not isinstance(state_hub, StateHub):
            raise TypeError("state_hub must be a StateHub")
        if not isinstance(artifact_store, ArtifactStore):
            raise TypeError("artifact_store must be an ArtifactStore")
        if not isinstance(memory_service, MemoryService):
            raise TypeError("memory_service must be a MemoryService")

        self._state_hub = state_hub
        self._artifact_store = artifact_store
        self._memory_service = memory_service

    def resolve(
        self,
        ref: StateRef | ArtifactRef | MemoryRef,
    ) -> object:
        if isinstance(ref, StateRef):
            return self.resolve_state(ref)
        if isinstance(ref, ArtifactRef):
            return self.resolve_artifact(ref)
        if isinstance(ref, MemoryRef):
            return self.resolve_memory(ref)
        raise TypeError("ref must be a StateRef, ArtifactRef, or MemoryRef")

    def resolve_state(
        self,
        ref: StateRef,
    ) -> np.ndarray:
        if not isinstance(ref, StateRef):
            raise TypeError("ref must be a StateRef")
        return self._state_hub.resolve_array(ref)

    def resolve_artifact(
        self,
        ref: ArtifactRef,
    ) -> object:
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.media_type == "application/json":
            return self._artifact_store.get_json(ref)
        return self._artifact_store.get_bytes(ref)

    def resolve_memory(
        self,
        ref: MemoryRef,
    ) -> MemoryRecord:
        if not isinstance(ref, MemoryRef):
            raise TypeError("ref must be a MemoryRef")
        record = self._memory_service.get(ref.memory_id)
        if record is None:
            raise KeyError(ref.memory_id)
        return record
