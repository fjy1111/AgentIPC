from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from agentipc.artifacts.store import ArtifactStore
from agentipc.memory.models import MemoryRecord
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.runtime.reference_resolver import ReferenceResolver
from agentipc.state.hub import StateHub


def _memory_service(tmp_path: Path) -> MemoryService:
    return MemoryService(
        SQLiteMemoryStore(tmp_path / "memory"),
        HashEmbeddingProvider(dim=8),
        VectorIndex(8),
    )


def _resolver(
    tmp_path: Path,
) -> tuple[ReferenceResolver, StateHub, ArtifactStore, MemoryService]:
    state_hub = StateHub(transport="inproc")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    memory_service = _memory_service(tmp_path)
    return (
        ReferenceResolver(
            state_hub=state_hub,
            artifact_store=artifact_store,
            memory_service=memory_service,
        ),
        state_hub,
        artifact_store,
        memory_service,
    )


def _memory_record(memory_id: str = "memory-ref-test") -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        source_agent="summarizer",
        task_topic="openEuler network",
        summary="Reusable network result",
        memory_type="result",
        tags=["network"],
        keywords=["NetworkManager"],
        payload={"answer": "MEMORY-PAYLOAD-SENTINEL"},
    )


def test_constructor_preserves_dependencies_and_rejects_invalid_types(
    tmp_path: Path,
) -> None:
    resolver, state_hub, artifact_store, memory_service = _resolver(tmp_path)

    assert resolver._state_hub is state_hub
    assert resolver._artifact_store is artifact_store
    assert resolver._memory_service is memory_service

    with pytest.raises(TypeError, match="state_hub"):
        ReferenceResolver(
            state_hub=object(),  # type: ignore[arg-type]
            artifact_store=artifact_store,
            memory_service=memory_service,
        )
    with pytest.raises(TypeError, match="artifact_store"):
        ReferenceResolver(
            state_hub=state_hub,
            artifact_store=object(),  # type: ignore[arg-type]
            memory_service=memory_service,
        )
    with pytest.raises(TypeError, match="memory_service"):
        ReferenceResolver(
            state_hub=state_hub,
            artifact_store=artifact_store,
            memory_service=object(),  # type: ignore[arg-type]
        )


def test_real_state_ref_resolves_exact_native_array(tmp_path: Path) -> None:
    resolver, state_hub, _, _ = _resolver(tmp_path)
    array = np.array([1.25, -2.5, 3.75], dtype=np.float32)
    state_ref = state_hub.put_array(
        array,
        kind="test-vector",
        summary="state sentinel",
    )

    resolved = resolver.resolve(state_ref)

    assert isinstance(resolved, np.ndarray)
    assert resolved.shape == array.shape
    assert resolved.dtype == array.dtype
    assert np.array_equal(resolved, array)
    assert np.array_equal(resolver.resolve_state(state_ref), array)


def test_real_json_and_raw_artifacts_resolve_native_content(tmp_path: Path) -> None:
    resolver, _, artifact_store, _ = _resolver(tmp_path)
    artifact_value = {
        "document_id": "artifact-doc",
        "text": "ARTIFACT-CONTENT-SENTINEL",
        "unicode": "openEuler 网络",
    }
    json_ref = artifact_store.put_json(
        artifact_value,
        summary="different ref summary",
    )
    payload = b"\x00\x01agentipc"
    bytes_ref = artifact_store.put_bytes(
        payload,
        media_type="application/octet-stream",
        summary="binary",
    )

    resolved_json = resolver.resolve(json_ref)
    resolved_bytes = resolver.resolve(bytes_ref)

    assert isinstance(resolved_json, dict)
    assert resolved_json == artifact_value
    assert resolver.resolve_artifact(json_ref) == artifact_value
    assert isinstance(resolved_bytes, bytes)
    assert resolved_bytes == payload
    assert resolver.resolve_artifact(bytes_ref) == payload


def test_real_memory_ref_resolves_record_without_marking_used(tmp_path: Path) -> None:
    resolver, _, _, memory_service = _resolver(tmp_path)
    stored = memory_service.write(_memory_record())
    memory_ref = MemoryRef(
        memory_id=stored.memory_id,
        score=0.9,
        match_type="hybrid",
        summary="memory ref summary",
    )
    before = memory_service.get(stored.memory_id)
    assert before is not None

    resolved = resolver.resolve(memory_ref)
    after = memory_service.get(stored.memory_id)

    assert isinstance(resolved, MemoryRecord)
    assert resolved == stored
    assert resolver.resolve_memory(memory_ref) == stored
    assert after is not None
    assert (
        after.reuse_count,
        after.success_count,
        after.failure_count,
    ) == (
        before.reuse_count,
        before.success_count,
        before.failure_count,
    ) == (0, 0, 0)


def test_missing_memory_raises_key_error(tmp_path: Path) -> None:
    resolver, _, _, _ = _resolver(tmp_path)
    ref = MemoryRef(
        memory_id="missing-memory",
        score=0.1,
        match_type="hybrid",
        summary="missing",
    )

    with pytest.raises(KeyError) as exc_info:
        resolver.resolve(ref)

    assert exc_info.value.args == ("missing-memory",)


@pytest.mark.parametrize("value", [object(), "ref", None])
def test_generic_resolve_rejects_unsupported_values(
    tmp_path: Path,
    value: object,
) -> None:
    resolver, _, _, _ = _resolver(tmp_path)

    with pytest.raises(TypeError, match="StateRef"):
        resolver.resolve(value)  # type: ignore[arg-type]


def test_typed_resolvers_reject_wrong_ref_types(tmp_path: Path) -> None:
    resolver, _, _, _ = _resolver(tmp_path)

    with pytest.raises(TypeError, match="StateRef"):
        resolver.resolve_state(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ArtifactRef"):
        resolver.resolve_artifact(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="MemoryRef"):
        resolver.resolve_memory(object())  # type: ignore[arg-type]


def test_state_backend_error_propagates_without_fallback(tmp_path: Path) -> None:
    resolver, state_hub, _, _ = _resolver(tmp_path)
    array = np.array([5.0, 6.0], dtype=np.float32)
    ref = state_hub.put_array(array, kind="vector", summary="summary fallback")
    bad_ref = ref.model_copy(update={"checksum": "0" * 64})

    with pytest.raises(ValueError, match="checksum mismatch"):
        resolver.resolve(bad_ref)


def test_missing_artifact_backend_error_propagates(tmp_path: Path) -> None:
    resolver, _, _, _ = _resolver(tmp_path)
    digest = "a" * 64
    ref = ArtifactRef(
        uri=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type="application/octet-stream",
        size_bytes=1,
        summary="missing",
    )

    with pytest.raises(FileNotFoundError):
        resolver.resolve(ref)
