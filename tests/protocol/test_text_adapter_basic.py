import pytest

from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef
from agentipc.protocol.text_adapter import render


def make_envelope(**overrides) -> AgentEnvelope:
    data = {
        "message_id": "msg_text",
        "trace_id": "trace_text",
        "task_id": "task_text",
        "step_id": "step_1",
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


def test_basic_render_returns_string() -> None:
    rendered = render(make_envelope())

    assert isinstance(rendered, str)


def test_render_contains_required_information() -> None:
    rendered = render(make_envelope())

    expected_fragments = [
        "Protocol version: agentipc/0.1",
        "Message ID: msg_text",
        "Trace ID: trace_text",
        "Task ID: task_text",
        "Step ID: step_1",
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

    for fragment in expected_fragments:
        assert fragment in rendered


def test_render_is_stable() -> None:
    envelope = make_envelope()

    assert render(envelope) == render(envelope)


def test_dict_insertion_order_does_not_change_rendering() -> None:
    first = make_envelope(
        args={"z": 1, "a": 2},
        metrics={"z_metric": 3, "a_metric": 4},
    )
    second = make_envelope(
        args={"a": 2, "z": 1},
        metrics={"a_metric": 4, "z_metric": 3},
    )

    assert render(first) == render(second)
    assert 'Arguments: {"a":2,"z":1}' in render(first)
    assert 'Metrics: {"a_metric":4,"z_metric":3}' in render(first)


def test_render_preserves_unicode() -> None:
    rendered = render(
        make_envelope(args={"query": "检查 openEuler 服务状态"})
    )

    assert "检查 openEuler 服务状态" in rendered
    assert "\\u68c0" not in rendered


def test_none_fields_have_explicit_stable_representation() -> None:
    rendered = render(
        make_envelope(action=None, capability=None, result=None)
    )

    assert "Action: null" in rendered
    assert "Capability: null" in rendered
    assert "Result: null" in rendered


def test_optional_resolver_is_accepted_but_unused_without_refs() -> None:
    envelope = make_envelope()

    assert render(envelope, resolver=object()) == render(envelope)


def test_state_ref_is_rejected() -> None:
    envelope = make_envelope(
        state_refs=[
            StateRef(
                uri="shm://agentipc/state-1",
                kind="vector",
                shape=[2],
                dtype="float32",
                nbytes=8,
                checksum="checksum",
                transport="shm",
                summary="state",
            )
        ]
    )

    with pytest.raises(ValueError, match="reference materialization is not supported"):
        render(envelope)


def test_artifact_ref_is_rejected() -> None:
    envelope = make_envelope(
        artifact_refs=[
            ArtifactRef(
                uri="artifact://sha256/example",
                sha256="b" * 64,
                media_type="application/json",
                size_bytes=10,
                summary="artifact",
            )
        ]
    )

    with pytest.raises(ValueError, match="reference materialization is not supported"):
        render(envelope)


def test_memory_ref_is_rejected() -> None:
    envelope = make_envelope(
        memory_refs=[
            MemoryRef(
                memory_id="mem_1",
                score=0.8,
                match_type="hybrid",
                summary="memory",
            )
        ]
    )

    with pytest.raises(ValueError, match="reference materialization is not supported"):
        render(envelope)