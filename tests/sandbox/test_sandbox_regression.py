from agentipc.sandbox.models import SandboxResult
from agentipc.sandbox.python_runner import PythonSandbox


def test_sandbox_regression_normal_path() -> None:
    result = PythonSandbox().run(
        'import sys\nprint("sandbox-normal")\nprint("sandbox-warning", file=sys.stderr)',
        timeout_sec=2.0,
    )

    assert isinstance(result, SandboxResult)
    assert result.exit_code == 0
    assert result.stdout == "sandbox-normal\n"
    assert result.stderr == "sandbox-warning\n"
    assert result.timed_out is False
    assert result.duration_ms >= 0


def test_sandbox_regression_runtime_error_path() -> None:
    result = PythonSandbox().run(
        'raise RuntimeError("sandbox-regression-boom")',
        timeout_sec=2.0,
    )

    assert isinstance(result, SandboxResult)
    assert result.exit_code != 0
    assert result.stdout == ""
    assert "RuntimeError: sandbox-regression-boom" in result.stderr
    assert result.timed_out is False
    assert result.duration_ms >= 0


def test_sandbox_regression_timeout_path_and_recovers() -> None:
    sandbox = PythonSandbox()

    timed_out = sandbox.run(
        "while True:\n    pass\n",
        timeout_sec=0.3,
    )

    assert isinstance(timed_out, SandboxResult)
    assert timed_out.timed_out is True
    assert type(timed_out.exit_code) is int
    assert timed_out.exit_code != 0
    assert timed_out.duration_ms >= 0

    recovered = sandbox.run(
        'print("after-timeout")',
        timeout_sec=2.0,
    )

    assert isinstance(recovered, SandboxResult)
    assert recovered.exit_code == 0
    assert recovered.stdout == "after-timeout\n"
    assert recovered.stderr == ""
    assert recovered.timed_out is False
