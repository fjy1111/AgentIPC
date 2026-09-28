from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentipc.evaluation.metrics import MetricsCollector
from agentipc.runtime.result import RunResult


EXPECTED_FIELDS = {
    "task_id",
    "success",
    "answer",
    "error",
    "metrics",
    "trace_path",
}


def test_success_result_exact_fields() -> None:
    metrics = {
        "message_count": 8,
        "protocol_bytes": 1024,
        "memory_retrieved": 1,
        "memory_used": 1,
        "memory_effective": 1,
    }
    result = RunResult(
        task_id="task-success",
        success=True,
        answer="Network troubleshooting completed.",
        error=None,
        metrics=metrics,
        trace_path="results/run/trace.jsonl",
    )

    assert set(RunResult.model_fields) == EXPECTED_FIELDS
    assert result.task_id == "task-success"
    assert result.success is True
    assert result.answer == "Network troubleshooting completed."
    assert result.error is None
    assert result.metrics == metrics
    assert result.trace_path == "results/run/trace.jsonl"
    assert set(result.model_dump()) == EXPECTED_FIELDS
    assert set(result.model_dump(mode="json")) == EXPECTED_FIELDS


def test_failure_result_with_error_and_empty_answer() -> None:
    error = {
        "type": "RuntimeError",
        "message": "memory write failed",
    }
    result = RunResult(
        task_id="task-failure",
        success=False,
        answer="",
        error=error,
        metrics={"message_count": 8},
        trace_path=None,
    )

    assert result.success is False
    assert result.answer == ""
    assert result.error == error
    assert result.trace_path is None


def test_optional_error_and_trace_path_default_none() -> None:
    result = RunResult(
        task_id="task",
        success=True,
        answer="ok",
        metrics={},
    )

    assert result.error is None
    assert result.trace_path is None


def test_json_round_trip_preserves_nested_data() -> None:
    result = RunResult(
        task_id="task-round-trip",
        success=False,
        answer="",
        error={
            "type": "ValueError",
            "message": "bad result",
            "stage": "summarize",
            "details": {"retryable": False},
        },
        metrics={
            "message_count": 8,
            "nested": {
                "counts": [1, 2, 3],
                "enabled": True,
            },
        },
        trace_path="results/round-trip/trace.jsonl",
    )

    assert result.model_dump()["metrics"] == result.metrics
    assert result.model_dump(mode="json")["error"] == result.error

    payload = result.model_dump_json()
    restored = RunResult.model_validate_json(payload)

    assert restored == result


def test_real_metrics_snapshot_dump_is_accepted() -> None:
    collector = MetricsCollector()
    collector.increment("message_count", 8)
    collector.increment("memory_retrieved", 2)
    collector.increment("memory_used", 1)
    collector.increment("memory_effective", 1)
    metrics = collector.snapshot().model_dump(mode="json")

    result = RunResult(
        task_id="task-metrics",
        success=True,
        answer="ok",
        metrics=metrics,
    )

    assert result.metrics == metrics
    assert result.metrics["message_count"] == 8
    assert result.metrics["memory_retrieved"] == 2
    assert result.metrics["memory_used"] == 1
    assert result.metrics["memory_effective"] == 1
    assert set(result.metrics) == {
        "message_count",
        "text_chars",
        "text_tokens",
        "protocol_bytes",
        "state_transfer_count",
        "state_bytes",
        "artifact_ref_count",
        "memory_retrieved",
        "memory_used",
        "memory_effective",
        "memory_harmful",
        "tool_call_count",
        "repeated_tool_call_count",
        "latency_ms",
        "success",
    }


def test_unicode_round_trip() -> None:
    result = RunResult(
        task_id="task-unicode",
        success=False,
        answer="openEuler 网络恢复完成",
        error={"message": "网络连接失败"},
        metrics={"note": "共享记忆"},
    )

    restored = RunResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.answer == "openEuler 网络恢复完成"
    assert restored.error == {"message": "网络连接失败"}
    assert restored.metrics == {"note": "共享记忆"}


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        RunResult(
            task_id="task",
            success=True,
            answer="ok",
            metrics={},
            unknown="x",  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("task_id", "", id="empty-task-id"),
        pytest.param("task_id", 123, id="non-string-task-id"),
        pytest.param("success", 1, id="integer-success"),
        pytest.param("success", 0, id="zero-success"),
        pytest.param("success", "true", id="string-success-true"),
        pytest.param("success", "false", id="string-success-false"),
        pytest.param("answer", 123, id="non-string-answer"),
        pytest.param("error", [], id="non-dict-error"),
        pytest.param("metrics", [], id="non-dict-metrics"),
        pytest.param("trace_path", 123, id="non-string-trace-path"),
    ],
)
def test_invalid_field_types_are_rejected(field: str, value: object) -> None:
    values: dict[str, object] = {
        "task_id": "task",
        "success": True,
        "answer": "ok",
        "error": None,
        "metrics": {},
        "trace_path": None,
    }
    values[field] = value

    with pytest.raises(ValidationError):
        RunResult.model_validate(values)


@pytest.mark.parametrize(
    "missing",
    ["task_id", "success", "answer", "metrics"],
)
def test_required_fields_are_required(missing: str) -> None:
    values: dict[str, object] = {
        "task_id": "task",
        "success": True,
        "answer": "ok",
        "metrics": {},
    }
    values.pop(missing)

    with pytest.raises(ValidationError):
        RunResult.model_validate(values)
