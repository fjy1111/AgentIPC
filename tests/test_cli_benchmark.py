import json
from pathlib import Path
import socket

import pytest

from agentipc.cli import main


def _fail_connect(*args: object, **kwargs: object) -> None:
    raise AssertionError("mock benchmark must not access the network")


def _only_result_dir(root: Path) -> Path:
    children = list(root.iterdir())
    assert len(children) == 1
    assert children[0].is_dir()
    return children[0]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_benchmark_smoke_writes_complete_offline_result_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(socket.socket, "connect", _fail_connect)
    monkeypatch.setenv("OPENAI_API_KEY", "benchmark-secret-must-not-leak")
    results_root = tmp_path / "results"

    exit_code = main(
        [
            "benchmark",
            "--suite",
            "smoke",
            "--repeat",
            "1",
            "--seed",
            "42",
            "--provider",
            "mock",
            "--results-root",
            str(results_root),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    result_dir = _only_result_dir(results_root)
    assert str(result_dir) in captured.out
    assert "record count: 4" in captured.out

    expected_files = {"raw.jsonl", "summary.json", "report.md", "environment.json"}
    assert expected_files.issubset({path.name for path in result_dir.iterdir()})

    raw = _read_jsonl(result_dir / "raw.jsonl")
    assert len(raw) == 4
    assert {record["experiment"]["name"] for record in raw} == {"A", "B", "C", "D"}
    assert len({record["task"] for record in raw}) == 1
    assert len({record["task_hash"] for record in raw}) == 1
    assert {record["seed"] for record in raw} == {42}
    assert all(record["run_result"]["success"] is True for record in raw)

    summary = json.loads((result_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["total_records"] == 4
    assert summary["task_count"] == 1
    assert summary["seed_count"] == 1
    assert set(summary["experiments"]) == {"A", "B", "C", "D"}
    assert set(summary["derived"]) == {"B_vs_A", "C_vs_B", "D_vs_C"}

    report = (result_dir / "report.md").read_text(encoding="utf-8")
    for heading in (
        "# AgentIPC Benchmark Report",
        "## Communication",
        "## State",
        "## Memory",
        "## Latency",
    ):
        assert heading in report

    environment_text = (result_dir / "environment.json").read_text(encoding="utf-8")
    environment = json.loads(environment_text)
    assert environment["llm_provider"] == "mock"
    assert environment["embedding_provider"] == "hash"
    assert "benchmark-secret-must-not-leak" not in environment_text


def test_benchmark_repeat_uses_sequential_reproducible_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(socket.socket, "connect", _fail_connect)
    results_root = tmp_path / "results"

    exit_code = main(
        [
            "benchmark",
            "--repeat",
            "2",
            "--seed",
            "100",
            "--provider",
            "mock",
            "--results-root",
            str(results_root),
        ]
    )

    assert exit_code == 0
    raw = _read_jsonl(_only_result_dir(results_root) / "raw.jsonl")
    assert len(raw) == 8
    assert {record["seed"] for record in raw} == {100, 101}
    for seed in (100, 101):
        records = [record for record in raw if record["seed"] == seed]
        assert len(records) == 4
        assert {record["experiment"]["name"] for record in records} == {
            "A",
            "B",
            "C",
            "D",
        }
        assert len({record["task_hash"] for record in records}) == 1


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--repeat", "0"), ("--seed", "-1")],
)
def test_benchmark_rejects_invalid_numeric_arguments(flag: str, value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["benchmark", flag, value])

    assert exc_info.value.code != 0
