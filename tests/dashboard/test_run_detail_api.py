"""Tests for GET /api/runs/{run_id}."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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
        "B_vs_A": DerivedMetrics(
            token_saving_rate=0.25,
            char_saving_rate=0.2,
            latency_improvement_rate=0.1,
            repeat_work_reduction_rate=None,
            effective_hit_rate=None,
        ),
        "C_vs_B": DerivedMetrics(
            token_saving_rate=0.1,
            char_saving_rate=0.1,
            latency_improvement_rate=0.05,
            repeat_work_reduction_rate=None,
            effective_hit_rate=None,
        ),
        "D_vs_C": DerivedMetrics(
            token_saving_rate=0.15,
            char_saving_rate=0.15,
            latency_improvement_rate=0.2,
            repeat_work_reduction_rate=0.5,
            effective_hit_rate=0.8,
        ),
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
        dependencies={"pydantic": "2.x"},
    )


def _write_ready_run(root: Path, run_id: str) -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    write_summary(run_dir / "summary.json", _summary())
    (run_dir / "environment.json").write_text(
        json.dumps(_environment().model_dump(mode="json")), encoding="utf-8"
    )
    (run_dir / "report.md").write_text("# AgentIPC Benchmark Report\n", encoding="utf-8")
    return run_dir


def test_ready_run_detail_contract(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_id = "20260930T120000000000Z-benchmark-smoke"
    run_dir = _write_ready_run(root, run_id)
    client = TestClient(create_app(root))

    response = client.get(f"/api/runs/{run_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == run_id
    assert body["summary"]["total_records"] == 4
    assert body["summary"]["task_count"] == 1
    assert body["summary"]["seed_count"] == 1
    assert set(body["summary"]["experiments"]) == {"A", "B", "C", "D"}
    assert "derived" not in body["summary"]
    assert body["derived"]["B_vs_A"]["token_saving_rate"] == 0.25
    assert body["derived"]["C_vs_B"]["token_saving_rate"] == 0.1
    assert body["derived"]["D_vs_C"]["effective_hit_rate"] == 0.8
    assert body["environment"]["llm_provider"] == "mock"
    assert body["environment"]["embedding_provider"] == "hash"
    assert body["report"]["title"] == "AgentIPC Benchmark Report"
    assert body["report"]["size_bytes"] == len((run_dir / "report.md").read_bytes())
    assert str(tmp_path) not in response.text


def test_unknown_incomplete_and_invalid_runs_are_404(tmp_path: Path) -> None:
    root = tmp_path / "results"
    incomplete_id = "20260929T120000000000Z-scenario-knowledge"
    invalid_id = "20260928T120000000000Z-benchmark-bad"
    (root / incomplete_id).mkdir(parents=True)
    bad = _write_ready_run(root, invalid_id)
    (bad / "environment.json").write_text("{}", encoding="utf-8")
    client = TestClient(create_app(root))

    for run_id in ["unknown", incomplete_id, invalid_id]:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 404
        assert response.json() == {"detail": "run not found"}


@pytest.mark.parametrize(
    "encoded_run_id",
    ["..%2Foutside", "a%2Fb", "a%5Cb", "%2Ftmp%2Fx"],
)
def test_encoded_path_traversal_cannot_read_outside_root(tmp_path: Path, encoded_run_id: str) -> None:
    root = tmp_path / "results"
    outside_id = "outside"
    _write_ready_run(tmp_path, outside_id)
    client = TestClient(create_app(root))

    response = client.get(f"/api/runs/{encoded_run_id}")

    assert response.status_code == 404
