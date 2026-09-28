from copy import deepcopy
from types import SimpleNamespace

import pytest

from agentipc.agents.base import BaseAgent
from agentipc.agents.executor import ExecutorAgent
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef


class PoisonService:
    def __getattr__(self, name: str):
        raise AssertionError(f"T084 must not access context service: {name}")


def _context():
    poison = PoisonService()
    return SimpleNamespace(
        provider_bundle=poison,
        state_hub=poison,
        memory_service=poison,
        artifact_store=poison,
        metrics=poison,
        trace_logger=poison,
    )


def _request(operation: object) -> AgentEnvelope:
    return AgentEnvelope(
        message_id="msg-execute-request",
        trace_id="trace-execute",
        task_id="task-execute",
        step_id="step-execute",
        sender="runtime",
        receiver="executor",
        message_type=MessageType.REQUEST,
        action=ActionType.EXECUTE,
        args={"operation": operation},
        metrics={"sentinel": "unchanged"},
    )


def test_identity_happy_path_is_protocol_valid_and_immutable() -> None:
    value = {
        "status": "checked",
        "nested": [1, 2.5, True, None, {"key": "值"}],
    }
    request = _request({"name": "identity", "value": value})
    before = request.model_copy(deep=True)
    operation_before = deepcopy(request.args["operation"])

    agent = ExecutorAgent()
    output = agent.handle(request, _context())

    assert isinstance(agent, BaseAgent)
    assert ExecutorAgent.agent_id == "executor"
    assert ExecutorAgent.capabilities == []
    assert output is not request
    assert output.trace_id == request.trace_id
    assert output.task_id == request.task_id
    assert output.step_id == request.step_id
    assert output.sender == "executor"
    assert output.receiver == "runtime"
    assert output.message_type is MessageType.RESULT
    assert output.action is ActionType.EXECUTE
    assert output.status is MessageStatus.OK
    assert output.capability == "execute"
    assert output.args == {}
    assert output.result == {
        "execution": {
            "operation": "identity",
            "output": value,
        }
    }
    assert output.state_refs == []
    assert output.artifact_refs == []
    assert output.memory_refs == []
    assert request == before
    assert request.args["operation"] == operation_before
    assert decode(encode(output)) == output


@pytest.mark.parametrize(
    ("operator", "operands", "expected"),
    [
        ("add", [2, 3], 5),
        ("subtract", [10, 4], 6),
        ("multiply", [3, 4], 12),
        ("divide", [7, 2], 3.5),
    ],
)
def test_arithmetic_operations(operator, operands, expected) -> None:
    operation = {
        "name": "arithmetic",
        "operator": operator,
        "operands": operands,
    }
    before = deepcopy(operation)

    output = ExecutorAgent().handle(_request(operation), _context())

    assert output.result == {
        "execution": {
            "operation": "arithmetic",
            "operator": operator,
            "output": expected,
        }
    }
    assert operation == before
    assert decode(encode(output)) == output


def test_json_pick_happy_path_preserves_requested_order_and_input() -> None:
    value = {
        "first": {"nested": [1, 2]},
        "second": "keep",
        "third": 3,
    }
    operation = {
        "name": "json_pick",
        "value": value,
        "keys": ["third", "first"],
    }
    before = deepcopy(operation)

    output = ExecutorAgent().handle(_request(operation), _context())

    execution = output.result["execution"]
    assert execution == {
        "operation": "json_pick",
        "output": {
            "third": 3,
            "first": {"nested": [1, 2]},
        },
    }
    assert list(execution["output"]) == ["third", "first"]
    assert operation == before


def test_output_does_not_echo_input_refs() -> None:
    request = _request({"name": "identity", "value": "ok"})
    request.state_refs = [
        StateRef(
            uri="inproc://agentipc/state-test",
            kind="test",
            shape=[1],
            dtype="<f4",
            nbytes=4,
            checksum="abc",
            transport="inproc",
            summary="state",
        )
    ]
    request.artifact_refs = [
        ArtifactRef(
            uri="artifact://sha256/" + "a" * 64,
            sha256="a" * 64,
            media_type="application/json",
            size_bytes=1,
            summary="artifact",
        )
    ]
    request.memory_refs = [
        MemoryRef(
            memory_id="mem-1",
            score=1.0,
            match_type="test",
            summary="memory",
        )
    ]
    before = request.model_copy(deep=True)

    output = ExecutorAgent().handle(request, _context())

    assert output.state_refs == []
    assert output.artifact_refs == []
    assert output.memory_refs == []
    assert request == before


def test_rejects_wrong_message_type() -> None:
    request = _request({"name": "identity", "value": 1}).model_copy(
        update={"message_type": MessageType.RESULT}
    )
    with pytest.raises(ValueError):
        ExecutorAgent().handle(request, _context())


def test_rejects_wrong_action() -> None:
    request = _request({"name": "identity", "value": 1}).model_copy(
        update={"action": ActionType.PLAN}
    )
    with pytest.raises(ValueError):
        ExecutorAgent().handle(request, _context())


def test_rejects_missing_operation() -> None:
    request = _request({"name": "identity", "value": 1})
    request.args.pop("operation")
    with pytest.raises(ValueError):
        ExecutorAgent().handle(request, _context())


@pytest.mark.parametrize("operation", [None, [], "identity", 1, True])
def test_operation_must_be_exact_dict(operation: object) -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(_request(operation), _context())


def test_rejects_missing_operation_name() -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(_request({"value": 1}), _context())


@pytest.mark.parametrize("name", [None, 1, True, []])
def test_operation_name_must_be_string(name: object) -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(_request({"name": name}), _context())


def test_operation_name_must_be_non_empty() -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(_request({"name": ""}), _context())


def test_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(_request({"name": "shell"}), _context())


@pytest.mark.parametrize(
    "operation",
    [
        {"name": "identity", "value": 1, "extra": True},
        {
            "name": "arithmetic",
            "operator": "add",
            "operands": [1, 2],
            "extra": True,
        },
        {"name": "json_pick", "value": {"a": 1}, "keys": ["a"], "extra": 1},
    ],
)
def test_operations_reject_unknown_fields(operation: dict) -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(_request(operation), _context())


@pytest.mark.parametrize(
    "operation",
    [
        {"name": "identity"},
        {"name": "arithmetic", "operands": [1, 2]},
        {"name": "arithmetic", "operator": "add"},
        {"name": "json_pick", "keys": ["a"]},
        {"name": "json_pick", "value": {"a": 1}},
    ],
)
def test_operation_specific_required_fields(operation: dict) -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(_request(operation), _context())


@pytest.mark.parametrize("value", [object(), {1, 2}, b"bytes", (1, 2)])
def test_identity_rejects_non_json_compatible_values(value: object) -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(
            _request({"name": "identity", "value": value}),
            _context(),
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_identity_rejects_non_finite_json_values(value: float) -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(
            _request({"name": "identity", "value": value}),
            _context(),
        )


def test_identity_rejects_non_string_json_object_keys() -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(
            _request({"name": "identity", "value": {1: "x"}}),
            _context(),
        )


@pytest.mark.parametrize("operator", ["power", "", None, True, 1])
def test_arithmetic_rejects_invalid_operator(operator: object) -> None:
    operation = {"name": "arithmetic", "operator": operator, "operands": [1, 2]}
    error = TypeError if type(operator) is not str else ValueError
    with pytest.raises(error):
        ExecutorAgent().handle(_request(operation), _context())


@pytest.mark.parametrize("operands", [(1, 2), "12", None, 3])
def test_arithmetic_operands_must_be_exact_list(operands: object) -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(
            _request({"name": "arithmetic", "operator": "add", "operands": operands}),
            _context(),
        )


@pytest.mark.parametrize("operands", [[], [1], [1, 2, 3]])
def test_arithmetic_requires_exactly_two_operands(operands: list) -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(
            _request({"name": "arithmetic", "operator": "add", "operands": operands}),
            _context(),
        )


@pytest.mark.parametrize("bad", [True, False, "1", None, [], {}])
def test_arithmetic_rejects_non_numeric_operands(bad: object) -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(
            _request({"name": "arithmetic", "operator": "add", "operands": [bad, 1]}),
            _context(),
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_arithmetic_rejects_non_finite_operands(bad: float) -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(
            _request({"name": "arithmetic", "operator": "add", "operands": [bad, 1]}),
            _context(),
        )


@pytest.mark.parametrize("zero", [0, 0.0, -0.0])
def test_arithmetic_rejects_divide_by_zero(zero: object) -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(
            _request({"name": "arithmetic", "operator": "divide", "operands": [1, zero]}),
            _context(),
        )


def test_arithmetic_rejects_non_finite_result() -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(
            _request(
                {
                    "name": "arithmetic",
                    "operator": "multiply",
                    "operands": [1e308, 1e308],
                }
            ),
            _context(),
        )


@pytest.mark.parametrize("value", [None, [], "object", 1, True])
def test_json_pick_value_must_be_exact_dict(value: object) -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(
            _request({"name": "json_pick", "value": value, "keys": []}),
            _context(),
        )


def test_json_pick_value_requires_string_keys() -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(
            _request({"name": "json_pick", "value": {1: "x"}, "keys": []}),
            _context(),
        )


@pytest.mark.parametrize("keys", [None, (), "a", 1, True])
def test_json_pick_keys_must_be_exact_list(keys: object) -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(
            _request({"name": "json_pick", "value": {"a": 1}, "keys": keys}),
            _context(),
        )


def test_json_pick_rejects_non_string_key() -> None:
    with pytest.raises(TypeError):
        ExecutorAgent().handle(
            _request({"name": "json_pick", "value": {"a": 1}, "keys": [1]}),
            _context(),
        )


def test_json_pick_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(
            _request({"name": "json_pick", "value": {"": 1}, "keys": [""]}),
            _context(),
        )


def test_json_pick_rejects_duplicate_key() -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(
            _request({"name": "json_pick", "value": {"a": 1}, "keys": ["a", "a"]}),
            _context(),
        )


def test_json_pick_rejects_missing_requested_key() -> None:
    with pytest.raises(ValueError):
        ExecutorAgent().handle(
            _request({"name": "json_pick", "value": {"a": 1}, "keys": ["b"]}),
            _context(),
        )
