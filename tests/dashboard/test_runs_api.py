"""Tests for GET /api/runs."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from agentipc.dashboard.app import create_app
from agentipc.evaluation.derived import DerivedMetrics
from agentipc.evaluation.env import EnvironmentSnapshot
from agentipc.evaluation.experiment import ExperimentName
from agentipc.evaluation.io import BenchmarkSummary, ExperimentSummary, write_summary


def _summary() -> BenchmarkSummary:
    experiments = {
        name: ExperimentSummary(
            experiment=ExperimentName(name),
            run_count=1,
            success_count=1,
            failure_count=0,
            success_rate=1.0,
            metrics={},
        )
        for name in ("A", "B", "C", "D")
    }
    derived = {
        name: DerivedMetrics(
            token_saving_rate=0.1,
            char_saving_rate=0.1,
            latency_improvement_rate=0.1,
            repeat_work_reduction_rate=None,
            effective_hit_rate=None,
        )
        for name in ("B_vs_A", "C_vs_B", "D_vs_C")
    }
    return BenchmarkSummary(
        total_records=4,
        task_count=1,
        seed_count=1,
        experiments=experiments,
        derived=derived,
    )


def _environment() -> EnvironmentSnapshot:
    return EnvironmentSnapshot(
        os_name="posix",
        platform="Linux-test",
        python_version="3.11.0",
        python_implementation="CPython",
        llm_provider="mock",
        embedding_provider="hash",
        token_method="unavailable",
        dependencies={},
    )


def _write_ready_run(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    write_summary(run_dir / "summary.json", _summary())
    (run_dir / "environment.json").write_text(
        json.dumps(_environment().model_dump(mode="json")), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        "# AgentIPC Benchmark Report\n\nsecret test value\n", encoding="utf-8"
    )
    return run_dir


def test_runs_api_returns_ready_incomplete_and_invalid_without_leaks(tmp_path: Path) -> None:
    root = tmp_path / "results"
    ready_id = "20260930T120000000000Z-benchmark-smoke"
    incomplete_id = "20260929T120000000000Z-scenario-knowledge"
    invalid_id = "20260928T120000000000Z-benchmark-bad"

    ready_dir = _write_ready_run(root, ready_id)
    (root / incomplete_id).mkdir(parents=True)
    invalid_dir = _write_ready_run(root, invalid_id)
    (invalid_dir / "summary.json").write_text("{", encoding="utf-8")

    client = TestClient(create_app(root))
    response = client.get("/api/runs")

    assert response.status_code == 200
    body = response.json()
    assert len(body["runs"]) == 3

    by_id = {item["run_id"]: item for item in body["runs"]}
    assert by_id[ready_id]["status"] == "ready"
    assert by_id[incomplete_id]["status"] == "incomplete"
    assert by_id[invalid_id]["status"] == "invalid"

    ready = by_id[ready_id]
    assert ready["llm_provider"] == "mock"
    assert ready["embedding_provider"] == "hash"
    assert ready["total_records"] == 4
    assert ready["task_count"] == 1
    assert ready["seed_count"] == 1
    assert ready["report_title"] == "AgentIPC Benchmark Report"
    assert ready["report_bytes"] == len((ready_dir / "report.md").read_bytes())

    serialized = response.text
    assert str(tmp_path) not in serialized
    assert "secret test value" not in serialized
    assert "api_key" not in serialized.lower()
    assert "base_url" not in serialized.lower()
