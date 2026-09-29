from __future__ import annotations

import json
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, StrictBool

from agentipc.scenarios.models import CodeActTask


_REL_TOL = 1e-9
_ABS_TOL = 1e-9
_SUPPORTED_OPERATIONS = frozenset({"codeact", "identity"})


class CodeActEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: StrictBool
    expected_result: Any
    actual_result: Any = None
    error: str | None = None


def evaluate_codeact_execution(
    task: CodeActTask,
    execution: dict[str, object],
) -> CodeActEvaluation:
    if not isinstance(task, CodeActTask):
        raise TypeError("task must be a CodeActTask")
    if type(execution) is not dict:
        raise TypeError("execution must be a dict[str, object]")

    expected = task.expected.result
    output_error, output = _validate_execution_output(execution)
    if output_error is not None:
        return CodeActEvaluation(
            success=False,
            expected_result=expected,
            error=output_error,
        )
    assert output is not None

    if output["timed_out"] is True:
        return CodeActEvaluation(
            success=False,
            expected_result=expected,
            error="sandbox_timeout",
        )
    if output["exit_code"] != 0:
        return CodeActEvaluation(
            success=False,
            expected_result=expected,
            error="sandbox_failure",
        )

    try:
        actual = json.loads(
            output["stdout"],
            parse_constant=_reject_non_json_constant,
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return CodeActEvaluation(
            success=False,
            expected_result=expected,
            error="invalid_json_stdout",
        )

    if not _is_json_value(actual):
        return CodeActEvaluation(
            success=False,
            expected_result=expected,
            error="invalid_json_stdout",
        )

    if not _json_values_equal(expected, actual):
        return CodeActEvaluation(
            success=False,
            expected_result=expected,
            actual_result=actual,
            error="result_mismatch",
        )

    return CodeActEvaluation(
        success=True,
        expected_result=expected,
        actual_result=actual,
        error=None,
    )


def _validate_execution_output(
    execution: dict[str, object],
) -> tuple[str | None, dict[str, object] | None]:
    operation = execution.get("operation")
    if type(operation) is not str or operation not in _SUPPORTED_OPERATIONS:
        return "invalid_execution", None

    output = execution.get("output")
    if type(output) is not dict:
        return "invalid_execution", None

    required = {"exit_code", "stdout", "stderr", "timed_out", "duration_ms"}
    if not required.issubset(output):
        return "invalid_execution", None

    if type(output["exit_code"]) is not int:
        return "invalid_execution", None
    if type(output["stdout"]) is not str:
        return "invalid_execution", None
    if type(output["stderr"]) is not str:
        return "invalid_execution", None
    if type(output["timed_out"]) is not bool:
        return "invalid_execution", None

    duration_ms = output["duration_ms"]
    if isinstance(duration_ms, bool) or type(duration_ms) not in (int, float):
        return "invalid_execution", None
    if not math.isfinite(float(duration_ms)) or duration_ms < 0:
        return "invalid_execution", None

    return None, output


def _reject_non_json_constant(value: str) -> object:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _is_json_value(value: object) -> bool:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return True
    if value_type is float:
        return math.isfinite(value)
    if value_type is list:
        return all(_is_json_value(item) for item in value)
    if value_type is dict:
        return all(
            type(key) is str and _is_json_value(item)
            for key, item in value.items()
        )
    return False


def _json_values_equal(expected: object, actual: object) -> bool:
    expected_type = type(expected)
    actual_type = type(actual)

    if expected is None or expected_type in (str, bool):
        return actual_type is expected_type and actual == expected

    if expected_type in (int, float):
        if actual_type not in (int, float):
            return False
        if expected_type is float and not math.isfinite(expected):
            return False
        if actual_type is float and not math.isfinite(actual):
            return False
        try:
            return math.isclose(
                expected,
                actual,
                rel_tol=_REL_TOL,
                abs_tol=_ABS_TOL,
            )
        except OverflowError:
            return expected_type is int and actual_type is int and expected == actual

    if expected_type is list:
        if actual_type is not list or len(expected) != len(actual):
            return False
        return all(
            _json_values_equal(expected_item, actual_item)
            for expected_item, actual_item in zip(expected, actual, strict=True)
        )

    if expected_type is dict:
        if actual_type is not dict or set(expected) != set(actual):
            return False
        return all(
            _json_values_equal(expected[key], actual[key])
            for key in expected
        )

    return False
