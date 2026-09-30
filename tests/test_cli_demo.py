import socket

import pytest

from agentipc.cli import main
from agentipc.demo import run_demo
from agentipc.evaluation.experiment import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_D,
)
from agentipc.runtime.context import RunMode


def test_demo_programmatic_report_uses_fair_a_b_d_runtime() -> None:
    report = run_demo(provider="mock")

    assert len(report.records) == 3
    record_a, record_b, record_d = report.records

    assert record_a.experiment == EXPERIMENT_A
    assert record_b.experiment == EXPERIMENT_B
    assert record_d.experiment == EXPERIMENT_D

    assert record_a.task == record_b.task == record_d.task
    assert record_a.seed == record_b.seed == record_d.seed == 42
    assert all(record.run_result.success is True for record in report.records)
    assert all(record.run_result.error is None for record in report.records)
    assert (
        record_a.run_result.answer
        == record_b.run_result.answer
        == record_d.run_result.answer
    )

    assert record_a.experiment.mode is RunMode.TEXT
    assert record_a.experiment.use_state is False
    assert record_a.experiment.use_memory is False

    assert record_b.experiment.mode is RunMode.STRUCTURED
    assert record_b.experiment.use_state is False
    assert record_b.experiment.use_memory is False

    assert record_d.experiment.mode is RunMode.STRUCTURED
    assert record_d.experiment.use_state is True
    assert record_d.experiment.use_memory is True

    metrics_a = record_a.run_result.metrics
    metrics_b = record_b.run_result.metrics
    metrics_d = record_d.run_result.metrics

    assert metrics_a["text_chars"] > 0


    assert metrics_b["protocol_bytes"] > 0
    assert metrics_b["state_transfer_count"] == 0

    assert metrics_d["protocol_bytes"] > 0
    assert metrics_d["state_transfer_count"] > 0
    assert metrics_d["state_bytes"] > 0
    assert metrics_d["memory_used"] == 0

    assert all(
        record.run_result.metrics["tool_call_count"] == 0
        for record in report.records
    )


def test_demo_cli_mock_is_offline_and_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("mock demo must not access the network")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)

    exit_code = main(["demo", "--provider", "mock"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Experiment A" in captured.out
    assert "Experiment B" in captured.out
    assert "Experiment D" in captured.out
    assert "Text" in captured.out
    assert "Structured" in captured.out
    assert "not a benchmark conclusion" in captured.out


def test_demo_rejects_unsupported_provider() -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["demo", "--provider", "openai"])

    assert exc_info.value.code != 0
