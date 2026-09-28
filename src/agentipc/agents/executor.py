from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from agentipc.agents.base import BaseAgent
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType

if TYPE_CHECKING:
    from agentipc.runtime.context import RunContext


_IDENTITY_FIELDS = frozenset({"name", "value"})
_ARITHMETIC_FIELDS = frozenset({"name", "operator", "operands"})
_JSON_PICK_FIELDS = frozenset({"name", "value", "keys"})
_ARITHMETIC_OPERATORS = frozenset({"add", "subtract", "multiply", "divide"})


class ExecutorAgent(BaseAgent):
    agent_id = "executor"
    capabilities: list[str] = []

    def handle(
        self,
        envelope: AgentEnvelope,
        ctx: "RunContext",
    ) -> AgentEnvelope:
        del ctx

        if envelope.message_type is not MessageType.REQUEST:
            raise ValueError("executor only accepts REQUEST envelopes")
        if envelope.action is not ActionType.EXECUTE:
            raise ValueError("executor only accepts EXECUTE actions")

        if "operation" not in envelope.args:
            raise ValueError("executor requires args['operation']")
        operation = envelope.args["operation"]
        if type(operation) is not dict:
            raise TypeError("operation must be a dict")

        if "name" not in operation:
            raise ValueError("operation requires name")
        name = operation["name"]
        if type(name) is not str:
            raise TypeError("operation name must be a str")
        if name == "":
            raise ValueError("operation name must be non-empty")

        if name == "identity":
            execution = _identity(operation)
        elif name == "arithmetic":
            execution = _arithmetic(operation)
        elif name == "json_pick":
            execution = _json_pick(operation)
        else:
            raise ValueError(f"unsupported operation: {name!r}")

        return AgentEnvelope(
            trace_id=envelope.trace_id,
            task_id=envelope.task_id,
            step_id=envelope.step_id,
            sender=self.agent_id,
            receiver=envelope.sender,
            message_type=MessageType.RESULT,
            action=ActionType.EXECUTE,
            capability="execute",
            result={"execution": execution},
            status=MessageStatus.OK,
            state_refs=[],
            artifact_refs=[],
            memory_refs=[],
        )


def _identity(operation: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(operation, _IDENTITY_FIELDS, operation_name="identity")
    if "value" not in operation:
        raise ValueError("identity operation requires value")

    value = operation["value"]
    _validate_json_compatible(value, field_name="identity value")
    return {
        "operation": "identity",
        "output": value,
    }


def _arithmetic(operation: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(
        operation,
        _ARITHMETIC_FIELDS,
        operation_name="arithmetic",
    )

    if "operator" not in operation:
        raise ValueError("arithmetic operation requires operator")
    operator = operation["operator"]
    if type(operator) is not str:
        raise TypeError("arithmetic operator must be a str")
    if operator not in _ARITHMETIC_OPERATORS:
        raise ValueError(f"unsupported arithmetic operator: {operator!r}")

    if "operands" not in operation:
        raise ValueError("arithmetic operation requires operands")
    operands = operation["operands"]
    if type(operands) is not list:
        raise TypeError("arithmetic operands must be a list")
    if len(operands) != 2:
        raise ValueError("arithmetic requires exactly two operands")

    left, right = operands
    _validate_number(left, field_name="left operand")
    _validate_number(right, field_name="right operand")

    if operator == "add":
        output = left + right
    elif operator == "subtract":
        output = left - right
    elif operator == "multiply":
        output = left * right
    else:
        if right == 0:
            raise ValueError("division by zero is not allowed")
        try:
            output = left / right
        except OverflowError as exc:
            raise ValueError("arithmetic result must be finite") from exc

    if type(output) is float and not math.isfinite(output):
        raise ValueError("arithmetic result must be finite")
    _validate_json_compatible(output, field_name="arithmetic result")

    return {
        "operation": "arithmetic",
        "operator": operator,
        "output": output,
    }


def _json_pick(operation: dict[str, Any]) -> dict[str, Any]:
    _require_exact_fields(
        operation,
        _JSON_PICK_FIELDS,
        operation_name="json_pick",
    )

    if "value" not in operation:
        raise ValueError("json_pick operation requires value")
    value = operation["value"]
    if type(value) is not dict:
        raise TypeError("json_pick value must be a dict")
    if any(type(key) is not str for key in value):
        raise TypeError("json_pick value keys must be str")
    _validate_json_compatible(value, field_name="json_pick value")

    if "keys" not in operation:
        raise ValueError("json_pick operation requires keys")
    keys = operation["keys"]
    if type(keys) is not list:
        raise TypeError("json_pick keys must be a list")

    seen: set[str] = set()
    selected: dict[str, Any] = {}
    for key in keys:
        if type(key) is not str:
            raise TypeError("json_pick keys items must be str")
        if key == "":
            raise ValueError("json_pick keys items must be non-empty")
        if key in seen:
            raise ValueError(f"duplicate json_pick key: {key!r}")
        seen.add(key)
        if key not in value:
            raise ValueError(f"json_pick requested missing key: {key!r}")
        selected[key] = value[key]

    return {
        "operation": "json_pick",
        "output": selected,
    }


def _require_exact_fields(
    operation: dict[str, Any],
    allowed: frozenset[str],
    *,
    operation_name: str,
) -> None:
    unknown = set(operation) - allowed
    if unknown:
        fields = ", ".join(sorted(repr(field) for field in unknown))
        raise ValueError(
            f"{operation_name} operation contains unknown fields: {fields}"
        )


def _validate_number(value: object, *, field_name: str) -> None:
    if type(value) not in (int, float):
        raise TypeError(f"{field_name} must be an int or float")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _validate_json_compatible(value: object, *, field_name: str) -> None:
    _validate_json_types(value, field_name=field_name)
    try:
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        )
    except TypeError as exc:
        raise TypeError(f"{field_name} must be JSON-compatible") from exc
    except ValueError as exc:
        raise ValueError(f"{field_name} must be JSON-compatible") from exc


def _validate_json_types(value: object, *, field_name: str) -> None:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite floats")
        return
    if value_type is list:
        for item in value:
            _validate_json_types(item, field_name=field_name)
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{field_name} dict keys must be str")
            _validate_json_types(item, field_name=field_name)
        return
    raise TypeError(f"{field_name} must be JSON-compatible")
