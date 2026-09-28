from pathlib import Path
import pytest

from agentipc.sandbox.models import SandboxResult
from agentipc.sandbox.python_runner import PythonSandbox


def test_print_stdout_success() -> None:
    result = PythonSandbox().run('print("hello")', timeout_sec=2.0)
    assert isinstance(result, SandboxResult)
    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""
    assert result.timed_out is False
    assert result.duration_ms >= 0


def test_stderr_and_explicit_nonzero_exit() -> None:
    code = 'import sys\nprint("boom", file=sys.stderr)\nraise SystemExit(7)\n'
    result = PythonSandbox().run(code, timeout_sec=2.0)
    assert result.exit_code == 7
    assert result.stdout == ""
    assert result.stderr == "boom\n"
    assert result.timed_out is False


def test_runtime_exception_becomes_stderr_and_nonzero_result() -> None:
    result = PythonSandbox().run('raise RuntimeError("boom")', timeout_sec=2.0)
    assert result.exit_code != 0
    assert result.stdout == ""
    assert "RuntimeError: boom" in result.stderr
    assert result.timed_out is False


def test_unicode_stdout() -> None:
    result = PythonSandbox().run('print("你好，AgentIPC")', timeout_sec=2.0)
    assert result.exit_code == 0
    assert result.stdout == "你好，AgentIPC\n"


def test_each_invocation_has_independent_cwd_and_workspace_is_deleted() -> None:
    sandbox = PythonSandbox()
    first = sandbox.run(
        'from pathlib import Path\nPath("marker.txt").write_text("a")\nprint(Path.cwd())',
        timeout_sec=2.0,
    )
    workspace_path = Path(first.stdout.strip())
    assert first.exit_code == 0
    assert workspace_path.exists() is False

    second = sandbox.run(
        'from pathlib import Path\nprint(Path("marker.txt").exists())',
        timeout_sec=2.0,
    )
    assert second.exit_code == 0
    assert second.stdout == "False\n"


def test_parent_process_cwd_is_unchanged() -> None:
    before = Path.cwd()
    result = PythonSandbox().run('print("ok")', timeout_sec=2.0)
    assert result.exit_code == 0
    assert Path.cwd() == before


def test_empty_source_is_valid() -> None:
    result = PythonSandbox().run("", timeout_sec=2.0)
    assert result.exit_code == 0
    assert result.stdout == ""
    assert result.stderr == ""
    assert result.timed_out is False


def test_code_requires_exact_str_type() -> None:
    with pytest.raises(TypeError, match="code"):
        PythonSandbox().run(b'print("hello")', timeout_sec=2.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout_sec", [None, True, False])
def test_timeout_rejects_invalid_type(timeout_sec: object) -> None:
    with pytest.raises(TypeError, match="timeout_sec"):
        PythonSandbox().run("", timeout_sec=timeout_sec)  # type: ignore[arg-type]


@pytest.mark.parametrize("timeout_sec", [0, -1, float("nan"), float("inf")])
def test_timeout_rejects_invalid_value(timeout_sec: float) -> None:
    with pytest.raises(ValueError, match="timeout_sec"):
        PythonSandbox().run("", timeout_sec=timeout_sec)
