from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pytest

from agentipc.artifacts.store import ArtifactStore
from agentipc.memory.models import MemoryRecord
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.refs import MemoryRef
from agentipc.protocol.text_adapter import render
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.runtime.reference_resolver import ReferenceResolver
from agentipc.state.hub import StateHub


def _make_envelope(**overrides) -> AgentEnvelope:
    data = {
        "message_id": "msg_refs",
        "trace_id": "trace_refs",
        "task_id": "task_refs",
        "step_id": "step_refs",
        "sender": "planner",
        "receiver": "retriever",
        "message_type": MessageType.REQUEST,
        "action": ActionType.RETRIEVE,
        "capability": "retrieval",
        "args": {"query": "agent ipc"},
        "result": {"count": 2},
        "status": MessageStatus.OK,
        "created_at": 1_700_000_000.25,
        "metrics": {"latency_ms": 1.5},
    }
    data.update(overrides)
    return AgentEnvelope(**data)


def _resolver(
    tmp_path: Path,
) -> tuple[ReferenceResolver, StateHub, ArtifactStore, MemoryService]:
    state_hub = StateHub(transport="inproc")
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    memory_service = MemoryService(
        SQLiteMemoryStore(tmp_path / "memory"),
        HashEmbeddingProvider(dim=8),
        VectorIndex(8),
    )
    resolver = ReferenceResolver(
        state_hub=state_hub,
        artifact_store=artifact_store,
        memory_service=memory_service,
    )
    return resolver, state_hub, artifact_store, memory_service


def _stored_memory(memory_service: MemoryService) -> MemoryRecord:
    return memory_service.write(
        MemoryRecord(
            memory_id="memory-ref-render",
            source_agent="summarizer",
            task_topic="openEuler network",
            summary="stored memory summary",
            memory_type="result",
            tags=["network"],
            keywords=["NetworkManager"],
            payload={
                "answer": "MEMORY-MATERIALIZED-CONTENT-SENTINEL",
                "unicode": "openEuler 网络恢复",
            },
        )
    )


def test_ref_free_output_regression_and_unused_resolver() -> None:
    envelope = _make_envelope()

    expected = "\n".join(
        [
            "Protocol version: agentipc/0.1",
            "Message ID: msg_refs",
            "Trace ID: trace_refs",
            "Task ID: task_refs",
            "Step ID: step_refs",
            "From: planner",
            "To: retriever",
            "Message type: REQUEST",
            "Action: RETRIEVE",
            "Capability: retrieval",
            'Arguments: {"query":"agent ipc"}',
            'Result: {"count":2}',
            "Status: OK",
            "Created at: 1700000000.25",
            'Metrics: {"latency_ms":1.5}',
        ]
    )

    assert render(envelope) == expected
    assert render(envelope, resolver=object()) == expected


def test_refs_require_resolver_and_invalid_resolver_is_clear(tmp_path: Path) -> None:
    _, state_hub, _, _ = _resolver(tmp_path)
    state_ref = state_hub.put_array(
        np.array([1.0], dtype=np.float32),
        kind="vector",
        summary="state",
    )
    envelope = _make_envelope(state_refs=[state_ref])

    with pytest.raises(
        ValueError,
        match="reference materialization is not supported",
    ):
        render(envelope)

    with pytest.raises(TypeError, match="callable resolve"):
        render(envelope, resolver=object())


def test_combined_real_refs_materialize_content_metadata_order_and_unicode(
    tmp_path: Path,
) -> None:
    resolver, state_hub, artifact_store, memory_service = _resolver(tmp_path)
    state_array = np.array([11.25, -22.5, 33.75], dtype=np.float32)
    state_ref = state_hub.put_array(
        state_array,
        kind="test-vector",
        summary="state summary without sentinel values",
    )
    artifact_value = {
        "text": "ARTIFACT-MATERIALIZED-CONTENT-SENTINEL",
        "unicode": "openEuler 网络恢复",
    }
    artifact_ref = artifact_store.put_json(
        artifact_value,
        summary="artifact summary without payload sentinel",
    )
    stored = _stored_memory(memory_service)
    memory_ref = MemoryRef(
        memory_id=stored.memory_id,
        score=0.9,
        match_type="hybrid",
        summary="memory ref summary without payload sentinel",
    )
    envelope = _make_envelope(
        state_refs=[state_ref],
        artifact_refs=[artifact_ref],
        memory_refs=[memory_ref],
    )
    before = envelope.model_copy(deep=True)
    state_before = state_array.copy()
    artifact_before = dict(artifact_value)
    memory_before = stored.model_copy(deep=True)

    first = render(envelope, resolver=resolver)
    second = render(envelope, resolver=resolver)

    assert first == second
    assert "11.25" in first
    assert "-22.5" in first
    assert "33.75" in first
    assert "ARTIFACT-MATERIALIZED-CONTENT-SENTINEL" in first
    assert "MEMORY-MATERIALIZED-CONTENT-SENTINEL" in first
    assert "openEuler 网络恢复" in first
    assert "\\u7f51" not in first
    assert state_ref.uri in first
    assert artifact_ref.uri in first
    assert memory_ref.memory_id in first
    assert first.index("State references:") < first.index("Artifact references:")
    assert first.index("Artifact references:") < first.index("Memory references:")
    assert envelope == before
    assert np.array_equal(state_array, state_before)
    assert artifact_value == artifact_before
    assert stored == memory_before


def test_each_ref_resolved_exactly_once_and_list_order_preserved(
    tmp_path: Path,
) -> None:
    resolver, _, artifact_store, _ = _resolver(tmp_path)
    first_ref = artifact_store.put_json(
        {"ordinal": "FIRST-REF-SENTINEL"},
        summary="first",
    )
    second_ref = artifact_store.put_json(
        {"ordinal": "SECOND-REF-SENTINEL"},
        summary="second",
    )

    class CountingResolver:
        def __init__(self, delegate: ReferenceResolver) -> None:
            self.delegate = delegate
            self.calls: list[object] = []

        def resolve(self, ref: object) -> object:
            self.calls.append(ref)
            return self.delegate.resolve(ref)  # type: ignore[arg-type]

    counting = CountingResolver(resolver)
    rendered = render(
        _make_envelope(artifact_refs=[first_ref, second_ref]),
        resolver=counting,  # type: ignore[arg-type]
    )

    assert counting.calls == [first_ref, second_ref]
    assert rendered.index("FIRST-REF-SENTINEL") < rendered.index(
        "SECOND-REF-SENTINEL"
    )


def test_state_array_text_conversion_includes_dtype_shape_and_complex_parts(
    tmp_path: Path,
) -> None:
    resolver, state_hub, _, _ = _resolver(tmp_path)
    real_ref = state_hub.put_array(
        np.array([1.25, -2.5, 3.75], dtype=np.float32),
        kind="real",
        summary="real",
    )
    complex_ref = state_hub.put_array(
        np.array([1 + 2j, -3 + 4j], dtype=np.complex64),
        kind="complex",
        summary="complex",
    )

    rendered = render(
        _make_envelope(state_refs=[real_ref, complex_ref]),
        resolver=resolver,
    )

    assert f'"dtype":"{np.dtype(np.float32).str}"' in rendered
    assert '"shape":[3]' in rendered
    assert '"values":[1.25,-2.5,3.75]' in rendered
    assert f'"dtype":"{np.dtype(np.complex64).str}"' in rendered
    assert '"real":[1.0,-3.0]' in rendered
    assert '"imag":[2.0,4.0]' in rendered


def test_bytes_materialize_as_utf8_or_base64(tmp_path: Path) -> None:
    resolver, _, artifact_store, _ = _resolver(tmp_path)
    utf8_ref = artifact_store.put_bytes(
        "openEuler 网络".encode("utf-8"),
        media_type="text/plain",
        summary="utf8",
    )
    binary_payload = b"\xff\x00\x80agentipc"
    binary_ref = artifact_store.put_bytes(
        binary_payload,
        media_type="application/octet-stream",
        summary="binary",
    )

    rendered = render(
        _make_envelope(artifact_refs=[utf8_ref, binary_ref]),
        resolver=resolver,
    )

    assert '"encoding":"utf-8"' in rendered
    assert '"text":"openEuler 网络"' in rendered
    assert '"encoding":"base64"' in rendered
    assert base64.b64encode(binary_payload).decode("ascii") in rendered


def test_memory_record_is_json_materialized(tmp_path: Path) -> None:
    resolver, _, _, memory_service = _resolver(tmp_path)
    stored = _stored_memory(memory_service)
    ref = MemoryRef(
        memory_id=stored.memory_id,
        score=0.8,
        match_type="hybrid",
        summary="ref summary",
    )

    rendered = render(_make_envelope(memory_refs=[ref]), resolver=resolver)

    assert '"memory_type":"result"' in rendered
    assert '"payload":{"answer":"MEMORY-MATERIALIZED-CONTENT-SENTINEL"' in rendered


def test_resolver_errors_propagate_without_summary_fallback(tmp_path: Path) -> None:
    resolver, _, _, _ = _resolver(tmp_path)
    ref = MemoryRef(
        memory_id="missing-memory",
        score=0.4,
        match_type="hybrid",
        summary="SUMMARY-FALLBACK-MUST-NOT-BE-USED",
    )

    with pytest.raises(KeyError) as exc_info:
        render(_make_envelope(memory_refs=[ref]), resolver=resolver)

    assert exc_info.value.args == ("missing-memory",)
