import json

import pytest

from agentipc.scenarios.codeact_eval import evaluate_codeact_execution
from agentipc.scenarios.models import CodeActTask


def _task(expected: object) -> CodeActTask:
    return CodeActTask.model_validate(
        {
            "group_id": "codeact-eval",
            "round": 1,
            "description": "Evaluate deterministic CodeAct output.",
            "code": "print('unused')\n",
            "input_artifacts": ["data/input.txt"],
            "expected": {"result": expected},
            "reuse_hint": {
                "source_rounds": [],
                "topics": ["codeact"],
                "expected_reuse": False,
            },
        }
    )


def _execution(
    stdout: str,
    *,
    operation: str = "codeact",
    exit_code: int = 0,
    timed_out: bool = False,
) -> dict[str, object]:
    return {
        "operation": operation,
        "output": {
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": "",
            "timed_out": timed_out,
            "duration_ms": 1.25,
        },
    }


@pytest.mark.parametrize("value", [None, True, 7, 1.25, "ok"])
def test_scalar_pass(value: object) -> None:
    result = evaluate_codeact_execution(
        _task(value),
        _execution(json.dumps(value)),
    )

    assert result.success is True
    assert result.actual_result == value
    assert result.error is None


def test_list_pass() -> None:
    expected = [1, "two", False, None]
    result = evaluate_codeact_execution(
        _task(expected),
        _execution(json.dumps(expected)),
    )

    assert result.success is True


def test_dict_pass_and_key_order_does_not_matter() -> None:
    expected = {"a": 1, "b": 2}
    result = evaluate_codeact_execution(
        _task(expected),
        _execution('{"b":2,"a":1}'),
    )

    assert result.success is True


def test_nested_pass() -> None:
    expected = {"items": [1, {"ok": True, "ratio": 0.5}], "missing": None}
    result = evaluate_codeact_execution(
        _task(expected),
        _execution(json.dumps(expected)),
    )

    assert result.success is True


def test_float_within_tolerance_pass() -> None:
    result = evaluate_codeact_execution(
        _task(1.0),
        _execution("1.0000000005"),
    )

    assert result.success is True


def test_float_outside_tolerance_fail() -> None:
    result = evaluate_codeact_execution(
        _task(1.0),
        _execution("1.000001"),
    )

    assert result.success is False
    assert result.error == "result_mismatch"


def test_list_order_mismatch_fail() -> None:
    result = evaluate_codeact_execution(
        _task([1, 2, 3]),
        _execution("[3,2,1]"),
    )

    assert result.success is False
    assert result.error == "result_mismatch"


@pytest.mark.parametrize(
    ("expected", "stdout"),
    [
        (True, "1"),
        (False, "0"),
        (1, "true"),
        (0, "false"),
    ],
)
def test_bool_and_int_are_distinct(expected: object, stdout: str) -> None:
    result = evaluate_codeact_execution(_task(expected), _execution(stdout))

    assert result.success is False
    assert result.error == "result_mismatch"


@pytest.mark.parametrize("stdout", ["not-json", "1 2", "NaN", "Infinity"])
def test_invalid_json_stdout_fail(stdout: str) -> None:
    result = evaluate_codeact_execution(_task(1), _execution(stdout))

    assert result.success is False
    assert result.error == "invalid_json_stdout"


def test_non_zero_exit_fail() -> None:
    result = evaluate_codeact_execution(
        _task(1),
        _execution("1", exit_code=2),
    )

    assert result.success is False
    assert result.error == "sandbox_failure"


def test_timeout_fail() -> None:
    result = evaluate_codeact_execution(
        _task(1),
        _execution("1", timed_out=True),
    )

    assert result.success is False
    assert result.error == "sandbox_timeout"


def test_codeact_execution_pass() -> None:
    result = evaluate_codeact_execution(
        _task({"value": 42}),
        _execution('{"value":42}', operation="codeact"),
    )

    assert result.success is True


def test_identity_wrapped_cached_codeact_output_pass() -> None:
    result = evaluate_codeact_execution(
        _task({"value": 42}),
        _execution('{"value":42}', operation="identity"),
    )

    assert result.success is True
