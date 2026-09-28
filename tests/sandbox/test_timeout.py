from agentipc.sandbox.python_runner import PythonSandbox


def test_timeout_terminates_process_and_returns_result() -> None:
    result = PythonSandbox().run("while True:\n    pass\n", timeout_sec=0.3)
    assert result.timed_out is True
    assert type(result.exit_code) is int
    assert result.exit_code != 0
    assert result.duration_ms >= 0


def test_timeout_preserves_flushed_partial_stdout_when_available() -> None:
    result = PythonSandbox().run(
        'print("started", flush=True)\nwhile True:\n    pass\n',
        timeout_sec=0.5,
    )
    assert result.timed_out is True
    assert result.exit_code != 0
    assert "started" in result.stdout
