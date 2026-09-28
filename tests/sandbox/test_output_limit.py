from agentipc.sandbox.python_runner import PythonSandbox


OUTPUT_LIMIT_BYTES = 64 * 1024
TRUNCATION_MARKER = "\n...[AgentIPC output truncated]...\n"


def _ascii_payload_bytes(value: str) -> int:
    payload = value.removesuffix(TRUNCATION_MARKER)
    return len(payload.encode("utf-8"))


def test_large_stdout_is_bounded_and_marked() -> None:
    result = PythonSandbox().run(
        'import sys\nsys.stdout.write("x" * 200_000)',
        timeout_sec=2.0,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stderr == ""
    assert result.stdout.count(TRUNCATION_MARKER) == 1
    assert _ascii_payload_bytes(result.stdout) <= OUTPUT_LIMIT_BYTES
    assert result.stdout.count("x") == OUTPUT_LIMIT_BYTES


def test_large_stderr_is_bounded_and_marked() -> None:
    result = PythonSandbox().run(
        'import sys\nsys.stderr.write("e" * 200_000)',
        timeout_sec=2.0,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout == ""
    assert result.stderr.count(TRUNCATION_MARKER) == 1
    assert _ascii_payload_bytes(result.stderr) <= OUTPUT_LIMIT_BYTES
    assert result.stderr.removesuffix(TRUNCATION_MARKER) == "e" * OUTPUT_LIMIT_BYTES


def test_large_stdout_and_stderr_are_drained_concurrently() -> None:
    code = (
        "import sys\n"
        'stdout_chunk = "o" * 4096\n'
        'stderr_chunk = "e" * 4096\n'
        "for _ in range(64):\n"
        "    sys.stdout.write(stdout_chunk)\n"
        "    sys.stdout.flush()\n"
        "    sys.stderr.write(stderr_chunk)\n"
        "    sys.stderr.flush()\n"
    )

    result = PythonSandbox().run(code, timeout_sec=3.0)

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout.count(TRUNCATION_MARKER) == 1
    assert result.stderr.count(TRUNCATION_MARKER) == 1
    assert _ascii_payload_bytes(result.stdout) <= OUTPUT_LIMIT_BYTES
    assert _ascii_payload_bytes(result.stderr) <= OUTPUT_LIMIT_BYTES


def test_below_limit_output_is_unchanged() -> None:
    result = PythonSandbox().run('print("small")', timeout_sec=2.0)

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout == "small\n"
    assert result.stderr == ""
    assert TRUNCATION_MARKER not in result.stdout
    assert TRUNCATION_MARKER not in result.stderr


def test_stdout_and_stderr_have_independent_budgets() -> None:
    result = PythonSandbox().run(
        'import sys\nsys.stdout.write("x" * 200_000)\nsys.stderr.write("important-error")',
        timeout_sec=2.0,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stdout.count(TRUNCATION_MARKER) == 1
    assert _ascii_payload_bytes(result.stdout) <= OUTPUT_LIMIT_BYTES
    assert result.stderr == "important-error"
    assert TRUNCATION_MARKER not in result.stderr


def test_unicode_truncation_boundary_decodes_safely() -> None:
    result = PythonSandbox().run(
        'import sys\nsys.stdout.write("你" * 100_000)',
        timeout_sec=2.0,
    )

    assert result.exit_code == 0
    assert result.timed_out is False
    assert result.stderr == ""
    assert result.stdout.count(TRUNCATION_MARKER) == 1


def test_timeout_with_continuous_large_output_remains_bounded() -> None:
    code = (
        "import sys\n"
        "while True:\n"
        '    sys.stdout.write("x" * 8192)\n'
        "    sys.stdout.flush()\n"
    )

    result = PythonSandbox().run(code, timeout_sec=1.5)

    assert result.timed_out is True
    assert result.exit_code != 0
    assert result.stdout.count(TRUNCATION_MARKER) == 1
    assert _ascii_payload_bytes(result.stdout) <= OUTPUT_LIMIT_BYTES


def test_runtime_error_remains_structured_result_without_false_truncation() -> None:
    result = PythonSandbox().run(
        'raise RuntimeError("boom")',
        timeout_sec=2.0,
    )

    assert result.exit_code != 0
    assert result.timed_out is False
    assert "RuntimeError: boom" in result.stderr
    assert TRUNCATION_MARKER not in result.stderr
