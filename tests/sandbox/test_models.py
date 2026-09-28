import math

import pytest
from pydantic import ValidationError

from agentipc.sandbox.models import SandboxResult


def make_result(**overrides) -> SandboxResult:
    data = {
        "exit_code": 0,
        "stdout": "hello\n",
        "stderr": "",
        "timed_out": False,
        "duration_ms": 12.5,
    }
    data.update(overrides)
    return SandboxResult(**data)


def test_sandbox_result_constructs_and_json_round_trips() -> None:
    result = make_result(duration_ms=12)
    restored = SandboxResult.model_validate_json(result.model_dump_json())
    assert restored == result
    assert type(restored.duration_ms) is float


def test_sandbox_result_allows_negative_exit_code_and_empty_output() -> None:
    result = make_result(
        exit_code=-9,
        stdout="",
        stderr="",
        timed_out=True,
        duration_ms=10.0,
    )
    assert result.exit_code == -9


def test_sandbox_result_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        make_result(unexpected=True)


def test_sandbox_result_rejects_bool_exit_code() -> None:
    with pytest.raises(ValidationError):
        make_result(exit_code=True)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stdout", 1),
        ("stderr", 1),
        ("timed_out", 1),
    ],
)
def test_sandbox_result_rejects_non_strict_field_values(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_result(**{field: value})


@pytest.mark.parametrize("duration_ms", [-1, float("nan"), float("inf"), float("-inf")])
def test_sandbox_result_rejects_invalid_duration(duration_ms: float) -> None:
    with pytest.raises(ValidationError):
        make_result(duration_ms=duration_ms)


def test_sandbox_result_duration_is_finite_and_nonnegative() -> None:
    result = make_result(duration_ms=0)
    assert result.duration_ms >= 0
    assert math.isfinite(result.duration_ms)
