"""Tests for Dashboard benchmark result repository scanning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentipc.dashboard.results import load_run_detail, scan_results
from agentipc.evaluation.derived import DerivedMetrics
from agentipc.evaluation.env import EnvironmentSnapshot
from agentipc.evaluation.experiment import ExperimentName
from agentipc.evaluation.io import BenchmarkSummary, ExperimentSummary, write_summary


def _summary(*, total_records: int = 4, task_count: int = 1, seed_count: int = 1) -> BenchmarkSummary:
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
        total_records=total_records,
        task_count=task_count,
        seed_count=seed_count,
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


def _write_ready_run(root: Path, run_id: str = "20260930T120000000000Z-benchmark-smoke") -> Path:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    write_summary(run_dir / "summary.json", _summary())
    (run_dir / "environment.json").write_text(
        json.dumps(_environment().model_dump(mode="json")), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        "# AgentIPC Benchmark Report\n\nsecret test value stays out of the index\n",
        encoding="utf-8",
    )
    return run_dir


def test_missing_results_root_is_empty(tmp_path: Path) -> None:
    assert scan_results(tmp_path / "missing") == []


def test_results_root_regular_file_is_configuration_error(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ValueError, match="results_root must be a directory"):
        scan_results(root)


def test_valid_benchmark_is_ready_with_metadata(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_dir = _write_ready_run(root)

    [entry] = scan_results(root)

    assert entry.status == "ready"
    assert entry.reason is None
    assert entry.llm_provider == "mock"
    assert entry.embedding_provider == "hash"
    assert entry.total_records == 4
    assert entry.task_count == 1
    assert entry.seed_count == 1
    assert entry.report_title == "AgentIPC Benchmark Report"
    assert entry.report_bytes == len((run_dir / "report.md").read_bytes())


def test_missing_report_is_incomplete(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_dir = _write_ready_run(root)
    (run_dir / "report.md").unlink()

    [entry] = scan_results(root)
    assert entry.status == "incomplete"
    assert entry.reason == "missing required files: report.md"


def test_invalid_summary_json_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_dir = _write_ready_run(root)
    (run_dir / "summary.json").write_text("{", encoding="utf-8")

    [entry] = scan_results(root)
    assert entry.status == "invalid"
    assert entry.reason == "summary.json malformed JSON"


def test_invalid_summary_schema_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_dir = _write_ready_run(root)
    (run_dir / "summary.json").write_text("{}", encoding="utf-8")

    [entry] = scan_results(root)
    assert entry.status == "invalid"
    assert entry.reason == "summary schema invalid"


@pytest.mark.parametrize("payload", ["{", "{}"])
def test_invalid_environment_json_or_schema_is_invalid(tmp_path: Path, payload: str) -> None:
    root = tmp_path / "results"
    run_dir = _write_ready_run(root)
    (run_dir / "environment.json").write_text(payload, encoding="utf-8")

    [entry] = scan_results(root)
    assert entry.status == "invalid"
    assert entry.reason in {
        "environment.json malformed JSON",
        "environment schema invalid",
    }


def test_malformed_report_title_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_dir = _write_ready_run(root)
    (run_dir / "report.md").write_text("not a markdown title\n", encoding="utf-8")

    [entry] = scan_results(root)
    assert entry.status == "invalid"
    assert entry.reason == "report title malformed"


def test_scenario_directory_is_incomplete_not_ready(tmp_path: Path) -> None:
    root = tmp_path / "results"
    run_dir = root / "20260930T120000000000Z-scenario-knowledge"
    run_dir.mkdir(parents=True)
    (run_dir / "scenario.json").write_text("{}", encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(_environment().model_dump(mode="json")), encoding="utf-8"
    )
    (run_dir / "raw.jsonl").write_text("", encoding="utf-8")

    [entry] = scan_results(root)
    assert entry.status == "incomplete"
    assert entry.reason == "missing required files: summary.json, report.md"


def test_direct_regular_files_are_ignored(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "README.txt").write_text("hello", encoding="utf-8")
    (root / "random.json").write_text("{}", encoding="utf-8")

    assert scan_results(root) == []


def test_runs_are_sorted_by_run_id_descending(tmp_path: Path) -> None:
    root = tmp_path / "results"
    for run_id in [
        "20260928T120000000000Z-benchmark-smoke",
        "20260930T120000000000Z-benchmark-smoke",
        "20260929T120000000000Z-benchmark-smoke",
    ]:
        _write_ready_run(root, run_id)

    assert [entry.run_id for entry in scan_results(root)] == [
        "20260930T120000000000Z-benchmark-smoke",
        "20260929T120000000000Z-benchmark-smoke",
        "20260928T120000000000Z-benchmark-smoke",
    ]


def test_detail_lookup_returns_only_ready_runs(tmp_path: Path) -> None:
    root = tmp_path / "results"
    ready_id = "20260930T120000000000Z-benchmark-smoke"
    incomplete_id = "20260929T120000000000Z-scenario-knowledge"
    invalid_id = "20260928T120000000000Z-benchmark-bad"

    _write_ready_run(root, ready_id)
    (root / incomplete_id).mkdir(parents=True)
    bad = _write_ready_run(root, invalid_id)
    (bad / "summary.json").write_text("{", encoding="utf-8")

    detail = load_run_detail(root, ready_id)
    assert detail is not None
    assert detail.run_id == ready_id
    assert detail.summary.total_records == 4

    assert load_run_detail(root, incomplete_id) is None
    assert load_run_detail(root, invalid_id) is None
    assert load_run_detail(root, "unknown") is None


@pytest.mark.parametrize("run_id", ["", ".", "..", "../outside", "a/b", "a\\b", "/tmp/x", "C:\\Temp\\x"])
def test_detail_lookup_rejects_path_traversal(tmp_path: Path, run_id: str) -> None:
    root = tmp_path / "results"
    _write_ready_run(root)
    outside = tmp_path / "outside"
    _write_ready_run(tmp_path, "outside")

    assert outside.exists()
    assert load_run_detail(root, run_id) is None
