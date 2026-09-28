from copy import deepcopy
from types import SimpleNamespace

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.sandbox.python_runner import PythonSandbox


def _request(operation: object) -> AgentEnvelope:
    return AgentEnvelope(
        message_id="msg-codeact-request",
        trace_id="trace-codeact",
        task_id="task-codeact",
        step_id="step-execute",
        sender="runtime",
        receiver="executor",
        message_type=MessageType.REQUEST,
        action=ActionType.EXECUTE,
        args={"operation": operation},
        metrics={"sentinel": "unchanged"},
    )


def _context(*, use_sandbox: bool) -> tuple[SimpleNamespace, MetricsCollector]:
    metrics = MetricsCollector()
    return (
        SimpleNamespace(
            use_sandbox=use_sandbox,
            metrics=metrics,
        ),
        metrics,
    )


def test_codeact_happy_path_is_protocol_valid_and_immutable() -> None:
    operation = {
        "name": "codeact",
        "code": 'print("hello-codeact")',
        "timeout_sec": 2.0,
    }
    request = _request(operation)
    before = request.model_copy(deep=True)
    operation_before = deepcopy(operation)
    ctx, metrics = _context(use_sandbox=True)

    response = ExecutorAgent().handle(request, ctx)

    assert response is not request
    assert response.trace_id == request.trace_id
    assert response.task_id == request.task_id
    assert response.step_id == request.step_id
    assert response.sender == "executor"
    assert response.receiver == request.sender
    assert response.message_type is MessageType.RESULT
    assert response.action is ActionType.EXECUTE
    assert response.capability == "execute"
    assert response.status is MessageStatus.OK
    assert response.state_refs == []
    assert response.artifact_refs == []
    assert response.memory_refs == []

    assert response.result is not None
    execution = response.result["execution"]
    assert execution["operation"] == "codeact"
    output = execution["output"]
    assert output["exit_code"] == 0
    assert output["stdout"] == "hello-codeact\n"
    assert output["stderr"] == ""
    assert output["timed_out"] is False
    assert output["duration_ms"] >= 0

    assert metrics.snapshot().tool_call_count == 1
    assert metrics.snapshot().repeated_tool_call_count == 0
    assert request == before
    assert request.args["operation"] == operation_before
    assert decode(encode(response)) == response


def test_codeact_python_runtime_error_is_structured_result() -> None:
    ctx, metrics = _context(use_sandbox=True)

    response = ExecutorAgent().handle(
        _request(
            {
                "name": "codeact",
                "code": 'raise RuntimeError("boom")',
                "timeout_sec": 2.0,
            }
        ),
        ctx,
    )

    assert response.status is MessageStatus.OK
    assert response.result is not None
    execution = response.result["execution"]
    assert execution["operation"] == "codeact"
    output = execution["output"]
    assert output["exit_code"] != 0
    assert "RuntimeError: boom" in output["stderr"]
    assert output["timed_out"] is False
    assert metrics.snapshot().tool_call_count == 1


def test_codeact_timeout_is_structured_result() -> None:
    ctx, metrics = _context(use_sandbox=True)

    response = ExecutorAgent().handle(
        _request(
            {
                "name": "codeact",
                "code": "while True:\n    pass\n",
                "timeout_sec": 0.3,
            }
        ),
        ctx,
    )

    assert response.status is MessageStatus.OK
    assert response.result is not None
    execution = response.result["execution"]
    assert execution["operation"] == "codeact"
    output = execution["output"]
    assert output["timed_out"] is True
    assert output["exit_code"] != 0
    assert metrics.snapshot().tool_call_count == 1


def test_codeact_rejects_sandbox_disabled_without_calling_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_run(*args, **kwargs):
        raise AssertionError("PythonSandbox.run must not be called")

    monkeypatch.setattr(PythonSandbox, "run", forbidden_run)
    ctx, metrics = _context(use_sandbox=False)

    with pytest.raises(ValueError, match="use_sandbox"):
        ExecutorAgent().handle(
            _request(
                {
                    "name": "codeact",
                    "code": 'print("disabled")',
                    "timeout_sec": 2.0,
                }
            ),
            ctx,
        )

    assert metrics.snapshot().tool_call_count == 0


@pytest.mark.parametrize(
    ("operation", "error_type"),
    [
        ({"name": "codeact", "timeout_sec": 2.0}, ValueError),
        ({"name": "codeact", "code": 'print("x")'}, ValueError),
        ({"name": "codeact", "code": "", "timeout_sec": 2.0}, ValueError),
        ({"name": "codeact", "code": None, "timeout_sec": 2.0}, TypeError),
        ({"name": "codeact", "code": b"print(1)", "timeout_sec": 2.0}, TypeError),
        ({"name": "codeact", "code": [], "timeout_sec": 2.0}, TypeError),
        ({"name": "codeact", "code": {}, "timeout_sec": 2.0}, TypeError),
        ({"name": "codeact", "code": (), "timeout_sec": 2.0}, TypeError),
        ({"name": "codeact", "code": 1, "timeout_sec": 2.0}, TypeError),
        ({"name": "codeact", "code": 1.5, "timeout_sec": 2.0}, TypeError),
        ({"name": "codeact", "code": True, "timeout_sec": 2.0}, TypeError),
        (
            {"name": "codeact", "code": 'print("x")', "timeout_sec": None},
            TypeError,
        ),
        (
            {"name": "codeact", "code": 'print("x")', "timeout_sec": "2"},
            TypeError,
        ),
        (
            {"name": "codeact", "code": 'print("x")', "timeout_sec": True},
            TypeError,
        ),
        (
            {"name": "codeact", "code": 'print("x")', "timeout_sec": False},
            TypeError,
        ),
        (
            {"name": "codeact", "code": 'print("x")', "timeout_sec": 0},
            ValueError,
        ),
        (
            {"name": "codeact", "code": 'print("x")', "timeout_sec": 0.0},
            ValueError,
        ),
        (
            {"name": "codeact", "code": 'print("x")', "timeout_sec": -1},
            ValueError,
        ),
        (
            {"name": "codeact", "code": 'print("x")', "timeout_sec": -0.1},
            ValueError,
        ),
        (
            {
                "name": "codeact",
                "code": 'print("x")',
                "timeout_sec": float("nan"),
            },
            ValueError,
        ),
        (
            {
                "name": "codeact",
                "code": 'print("x")',
                "timeout_sec": float("inf"),
            },
            ValueError,
        ),
        (
            {
                "name": "codeact",
                "code": 'print("x")',
                "timeout_sec": float("-inf"),
            },
            ValueError,
        ),
        (
            {
                "name": "codeact",
                "code": 'print("x")',
                "timeout_sec": 2.0,
                "extra": True,
            },
            ValueError,
        ),
    ],
)
def test_codeact_validation_fails_before_sandbox_or_metric_increment(
    operation: dict[str, object],
    error_type: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_run(*args, **kwargs):
        raise AssertionError("PythonSandbox.run must not be called")

    monkeypatch.setattr(PythonSandbox, "run", forbidden_run)
    ctx, metrics = _context(use_sandbox=True)

    with pytest.raises(error_type):
        ExecutorAgent().handle(_request(operation), ctx)

    assert metrics.snapshot().tool_call_count == 0


@pytest.mark.parametrize(
    "name",
    ["python", "python_exec", "shell", "bash", "script", "exec"],
)
def test_codeact_aliases_are_not_supported(name: str) -> None:
    ctx, metrics = _context(use_sandbox=True)

    with pytest.raises(ValueError, match="unsupported operation"):
        ExecutorAgent().handle(
            _request(
                {
                    "name": name,
                    "code": 'print("x")',
                    "timeout_sec": 2.0,
                }
            ),
            ctx,
        )

    assert metrics.snapshot().tool_call_count == 0


def test_legacy_identity_does_not_require_use_sandbox_and_does_not_count_tool() -> None:
    metrics = MetricsCollector()
    legacy_ctx = SimpleNamespace(metrics=metrics)

    response = ExecutorAgent().handle(
        _request({"name": "identity", "value": {"ok": True}}),
        legacy_ctx,
    )

    assert response.result == {
        "execution": {
            "operation": "identity",
            "output": {"ok": True},
        }
    }
    assert metrics.snapshot().tool_call_count == 0