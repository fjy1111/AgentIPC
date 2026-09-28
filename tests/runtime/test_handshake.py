from pathlib import Path

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.protocol.enums import (
    PROTOCOL_VERSION,
    MessageStatus,
    MessageType,
)
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.bootstrap import run_handshake


_AGENT_IDS = (
    "planner",
    "retriever",
    "executor",
    "summarizer",
)

_CAPABILITIES = {
    "planner": ["plan"],
    "retriever": ["retrieve"],
    "executor": ["execute"],
    "summarizer": ["summarize"],
}


def _registry_with_all_agents() -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent([]),
        ExecutorAgent(),
        SummarizerAgent(),
    ]:
        registry.register(agent)
    return registry


def _events(path: Path) -> list[TraceEvent]:
    if not path.exists():
        return []
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _assert_required_capabilities_empty(registry: CapabilityRegistry) -> None:
    for agent_id in _AGENT_IDS:
        assert registry.get(agent_id) is None


def test_handshake_emits_expected_control_flow_and_trace(tmp_path: Path) -> None:
    agent_registry = _registry_with_all_agents()
    capability_registry = CapabilityRegistry()
    trace_path = tmp_path / "handshake.jsonl"
    trace_logger = TraceLogger(trace_path)

    messages = run_handshake(
        agent_registry=agent_registry,
        capability_registry=capability_registry,
        trace_logger=trace_logger,
        trace_id="trace-handshake",
        task_id="task-handshake",
    )

    assert len(messages) == 12
    assert [message.message_type for message in messages] == [
        MessageType.HELLO,
        MessageType.REGISTER,
        MessageType.ACK,
    ] * 4
    assert len({message.message_id for message in messages}) == 12
    assert {message.trace_id for message in messages} == {"trace-handshake"}
    assert {message.task_id for message in messages} == {"task-handshake"}

    for group_index, agent_id in enumerate(_AGENT_IDS):
        hello, register, ack = messages[
            group_index * 3 : group_index * 3 + 3
        ]

        assert hello.sender == agent_id
        assert hello.receiver == "runtime"
        assert hello.step_id == f"handshake-{agent_id}-hello"

        assert register.sender == agent_id
        assert register.receiver == "runtime"
        assert register.step_id == f"handshake-{agent_id}-register"

        assert ack.sender == "runtime"
        assert ack.receiver == agent_id
        assert ack.step_id == f"handshake-{agent_id}-ack"
        assert ack.args == {"ack_message_id": register.message_id}
        assert ack.args["ack_message_id"] != hello.message_id

        payload = register.args["capability"]
        assert payload == {
            "agent_id": agent_id,
            "capabilities": _CAPABILITIES[agent_id],
            "protocol_versions": [PROTOCOL_VERSION],
            "metadata": {},
        }

        registry_record = capability_registry.get(agent_id)
        assert registry_record is not None
        assert registry_record.agent_id == agent_id
        assert registry_record.capabilities == _CAPABILITIES[agent_id]
        assert registry_record.protocol_versions == [PROTOCOL_VERSION]
        assert registry_record.metadata == {}

        assert hello.status is MessageStatus.PENDING
        assert register.status is MessageStatus.PENDING
        assert ack.status is MessageStatus.OK

    assert [record.agent_id for record in capability_registry.discover("plan")] == [
        "planner"
    ]
    assert [
        record.agent_id
        for record in capability_registry.discover("retrieve")
    ] == ["retriever"]
    assert [
        record.agent_id
        for record in capability_registry.discover("execute")
    ] == ["executor"]
    assert [
        record.agent_id
        for record in capability_registry.discover("summarize")
    ] == ["summarizer"]

    for message in messages:
        assert message.version == PROTOCOL_VERSION
        assert message.action is None
        assert message.state_refs == []
        assert message.artifact_refs == []
        assert message.memory_refs == []

    events = _events(trace_path)
    assert len(events) == 12
    assert [event.message_id for event in events] == [
        message.message_id for message in messages
    ]
    assert [event.message_type for event in events] == [
        message.message_type.value for message in messages
    ]
    assert {event.message_type for event in events} == {
        MessageType.HELLO.value,
        MessageType.REGISTER.value,
        MessageType.ACK.value,
    }
    assert sum(event.message_type == MessageType.HELLO.value for event in events) == 4
    assert sum(event.message_type == MessageType.REGISTER.value for event in events) == 4
    assert sum(event.message_type == MessageType.ACK.value for event in events) == 4
    assert all(event.action is None for event in events)
    assert all(event.trace_id == "trace-handshake" for event in events)
    assert all(event.task_id == "task-handshake" for event in events)
    assert all(event.state_refs == [] for event in events)
    assert all(event.artifact_refs == [] for event in events)
    assert all(event.memory_refs == [] for event in events)

    first_raw_event = trace_path.read_text(encoding="utf-8").splitlines()[0]
    assert '"args"' not in first_raw_event
    assert '"result"' not in first_raw_event
    assert '"metrics"' not in first_raw_event


def test_missing_agent_fails_atomically_without_trace(tmp_path: Path) -> None:
    agent_registry = AgentRegistry()
    for agent in [PlannerAgent(), RetrieverAgent([]), ExecutorAgent()]:
        agent_registry.register(agent)
    capability_registry = CapabilityRegistry()
    trace_path = tmp_path / "handshake.jsonl"

    with pytest.raises(KeyError):
        run_handshake(
            agent_registry=agent_registry,
            capability_registry=capability_registry,
            trace_logger=TraceLogger(trace_path),
            trace_id="trace-handshake",
            task_id="task-handshake",
        )

    _assert_required_capabilities_empty(capability_registry)
    assert _events(trace_path) == []


def test_invalid_agent_registry_fails_before_mutation(tmp_path: Path) -> None:
    capability_registry = CapabilityRegistry()
    trace_path = tmp_path / "handshake.jsonl"

    with pytest.raises(TypeError):
        run_handshake(
            agent_registry=object(),  # type: ignore[arg-type]
            capability_registry=capability_registry,
            trace_logger=TraceLogger(trace_path),
            trace_id="trace-handshake",
            task_id="task-handshake",
        )

    _assert_required_capabilities_empty(capability_registry)
    assert _events(trace_path) == []


def test_invalid_capability_registry_fails_before_trace(tmp_path: Path) -> None:
    trace_path = tmp_path / "handshake.jsonl"

    with pytest.raises(TypeError):
        run_handshake(
            agent_registry=_registry_with_all_agents(),
            capability_registry=object(),  # type: ignore[arg-type]
            trace_logger=TraceLogger(trace_path),
            trace_id="trace-handshake",
            task_id="task-handshake",
        )

    assert _events(trace_path) == []


def test_invalid_trace_logger_fails_before_capability_mutation() -> None:
    capability_registry = CapabilityRegistry()

    with pytest.raises(TypeError):
        run_handshake(
            agent_registry=_registry_with_all_agents(),
            capability_registry=capability_registry,
            trace_logger=object(),  # type: ignore[arg-type]
            trace_id="trace-handshake",
            task_id="task-handshake",
        )

    _assert_required_capabilities_empty(capability_registry)


@pytest.mark.parametrize(
    ("field", "value", "expected_exception"),
    [
        ("trace_id", "", ValueError),
        ("trace_id", 1, TypeError),
        ("trace_id", None, TypeError),
        ("task_id", "", ValueError),
        ("task_id", 1, TypeError),
        ("task_id", None, TypeError),
    ],
)
def test_invalid_identity_fails_before_mutation_or_trace(
    tmp_path: Path,
    field: str,
    value: object,
    expected_exception: type[Exception],
) -> None:
    capability_registry = CapabilityRegistry()
    trace_path = tmp_path / "handshake.jsonl"
    kwargs = {
        "agent_registry": _registry_with_all_agents(),
        "capability_registry": capability_registry,
        "trace_logger": TraceLogger(trace_path),
        "trace_id": "trace-handshake",
        "task_id": "task-handshake",
    }
    kwargs[field] = value

    with pytest.raises(expected_exception):
        run_handshake(**kwargs)  # type: ignore[arg-type]

    _assert_required_capabilities_empty(capability_registry)
    assert _events(trace_path) == []


def test_rehandshake_appends_new_messages(tmp_path: Path) -> None:
    agent_registry = _registry_with_all_agents()
    capability_registry = CapabilityRegistry()
    trace_path = tmp_path / "handshake.jsonl"
    trace_logger = TraceLogger(trace_path)

    first = run_handshake(
        agent_registry=agent_registry,
        capability_registry=capability_registry,
        trace_logger=trace_logger,
        trace_id="trace-handshake",
        task_id="task-handshake",
    )
    second = run_handshake(
        agent_registry=agent_registry,
        capability_registry=capability_registry,
        trace_logger=trace_logger,
        trace_id="trace-handshake",
        task_id="task-handshake",
    )

    assert len(first) == 12
    assert len(second) == 12
    assert len({message.message_id for message in first + second}) == 24

    events = _events(trace_path)
    assert len(events) == 24
    assert [event.message_id for event in events] == [
        message.message_id for message in first + second
    ]
