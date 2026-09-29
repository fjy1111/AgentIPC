"""Tests for summary JSON writer and BenchmarkSummary models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentipc.evaluation.derived import DerivedMetrics
from agentipc.evaluation.experiment import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    EXPERIMENT_D,
    ExperimentName,
)
from agentipc.evaluation.io import (
    BenchmarkSummary,
    ExperimentSummary,
    summarize_raw_records,
    write_summary,
)
from agentipc.evaluation.runner import RawRunRecord
from agentipc.evaluation.stats import AggregateStats
from agentipc.runtime.result import RunResult


def _build_raw_record(
    experiment_name: str,
    task: str,
    task_hash: str,
    seed: int,
    *,
    success: bool = True,
    text_tokens: int = 1000,
    text_chars: int = 5000,
    latency_ms: float = 200.0,
    repeated_tool_calls: int = 0,
    memory_used: int = 0,
    memory_effective: int = 0,
) -> RawRunRecord:
    """Build a RawRunRecord fixture with specified metrics."""
    metrics = {
        "message_count": 8,
        "text_chars": text_chars,
        "text_tokens": text_tokens,
        "protocol_bytes": 2000,
        "state_transfer_count": 0,
        "state_bytes": 0,
        "artifact_ref_count": 1,
        "memory_retrieved": memory_used,
        "memory_used": memory_used,
        "memory_effective": memory_effective,
        "memory_harmful": 0,
        "tool_call_count": 3,
        "repeated_tool_call_count": repeated_tool_calls,
        "llm_call_count": 4,
        "llm_prompt_tokens": 100,
        "llm_completion_tokens": 50,
        "llm_total_tokens": 150,
        "llm_usage_missing_count": 0,
        "llm_latency_ms": 50.0,
        "latency_ms": latency_ms,
        "success": success,
    }

    return RawRunRecord(
        experiment={"A": EXPERIMENT_A, "B": EXPERIMENT_B, "C": EXPERIMENT_C, "D": EXPERIMENT_D}[
            experiment_name
        ],
        task=task,
        task_hash=task_hash,
        seed=seed,
        use_sandbox=False,
        run_result=RunResult(
            task_id=f"task-{experiment_name}-{seed}",
            success=success,
            answer="mock answer" if success else "",
            error=None if success else {"type": "MockError", "message": "mock failure"},
            metrics=metrics,
            trace_path="/tmp/trace.jsonl",
        ),
    )


class TestExperimentSummaryModel:
    """Test ExperimentSummary data model."""

    def test_valid_summary_accepted(self):
        """Valid experiment summary is accepted."""
        summary = ExperimentSummary(
            experiment=ExperimentName.A,
            run_count=10,
            success_count=9,
            failure_count=1,
            success_rate=0.9,
            metrics={
                "text_tokens": AggregateStats(count=10, mean=1000.0, std=50.0, min=900.0, max=1100.0),
            },
        )

        assert summary.experiment == ExperimentName.A
        assert summary.run_count == 10
        assert summary.success_count == 9
        assert summary.failure_count == 1
        assert summary.success_rate == 0.9

    def test_success_rate_out_of_range_rejected(self):
        """success_rate > 1.0 is rejected."""
        with pytest.raises(ValidationError):
            ExperimentSummary(
                experiment=ExperimentName.A,
                run_count=10,
                success_count=11,
                failure_count=0,
                success_rate=1.1,
                metrics={},
            )


class TestBenchmarkSummaryModel:
    """Test BenchmarkSummary data model."""

    def test_valid_summary_accepted(self):
        """Valid benchmark summary is accepted."""
        summary = BenchmarkSummary(
            total_records=8,
            task_count=1,
            seed_count=2,
            experiments={
                "A": ExperimentSummary(
                    experiment=ExperimentName.A,
                    run_count=2,
                    success_count=2,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "B": ExperimentSummary(
                    experiment=ExperimentName.B,
                    run_count=2,
                    success_count=2,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "C": ExperimentSummary(
                    experiment=ExperimentName.C,
                    run_count=2,
                    success_count=2,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "D": ExperimentSummary(
                    experiment=ExperimentName.D,
                    run_count=2,
                    success_count=2,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
            },
            derived={
                "B_vs_A": DerivedMetrics(
                    token_saving_rate=0.25,
                    char_saving_rate=0.25,
                    latency_improvement_rate=0.10,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "C_vs_B": DerivedMetrics(
                    token_saving_rate=0.10,
                    char_saving_rate=0.10,
                    latency_improvement_rate=0.05,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "D_vs_C": DerivedMetrics(
                    token_saving_rate=0.15,
                    char_saving_rate=0.15,
                    latency_improvement_rate=0.20,
                    repeat_work_reduction_rate=0.50,
                    effective_hit_rate=0.75,
                ),
            },
        )

        assert summary.total_records == 8
        assert summary.task_count == 1
        assert summary.seed_count == 2
        assert len(summary.experiments) == 4
        assert len(summary.derived) == 3


class TestSummarizeRawRecords:
    """Test summarize_raw_records function."""

    def test_complete_abcd_summary_success(self):
        """Complete A/B/C/D records produce valid summary."""
        task_hash = "a" * 64
        records = []

        # A/B/C/D × 1 task × 2 seeds
        for exp_name in ["A", "B", "C", "D"]:
            for seed in [42, 43]:
                records.append(
                    _build_raw_record(
                        exp_name,
                        "task1",
                        task_hash,
                        seed,
                        text_tokens=1000,
                        text_chars=5000,
                    )
                )

        summary = summarize_raw_records(records)

        assert summary.total_records == 8
        assert summary.task_count == 1
        assert summary.seed_count == 2
        assert len(summary.experiments) == 4
        assert len(summary.derived) == 3

    def test_experiment_keys_complete(self):
        """Summary contains exactly A, B, C, D keys."""
        task_hash = "b" * 64
        records = []

        for exp_name in ["A", "B", "C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", task_hash, 42))

        summary = summarize_raw_records(records)

        assert set(summary.experiments.keys()) == {"A", "B", "C", "D"}

    def test_derived_keys_complete(self):
        """Summary contains exactly B_vs_A, C_vs_B, D_vs_C keys."""
        task_hash = "c" * 64
        records = []

        for exp_name in ["A", "B", "C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", task_hash, 42))

        summary = summarize_raw_records(records)

        assert set(summary.derived.keys()) == {"B_vs_A", "C_vs_B", "D_vs_C"}

    def test_success_count_correct(self):
        """Success and failure counts are correct."""
        task_hash = "d" * 64
        records = []

        # A: 2 success, 0 failure
        records.append(_build_raw_record("A", "task1", task_hash, 42, success=True))
        records.append(_build_raw_record("A", "task1", task_hash, 43, success=True))

        # B: 1 success, 1 failure
        records.append(_build_raw_record("B", "task1", task_hash, 42, success=True))
        records.append(_build_raw_record("B", "task1", task_hash, 43, success=False))

        # C: 0 success, 2 failure
        records.append(_build_raw_record("C", "task1", task_hash, 42, success=False))
        records.append(_build_raw_record("C", "task1", task_hash, 43, success=False))

        # D: 2 success, 0 failure
        records.append(_build_raw_record("D", "task1", task_hash, 42, success=True))
        records.append(_build_raw_record("D", "task1", task_hash, 43, success=True))

        summary = summarize_raw_records(records)

        assert summary.experiments["A"].success_count == 2
        assert summary.experiments["A"].failure_count == 0
        assert summary.experiments["A"].success_rate == 1.0

        assert summary.experiments["B"].success_count == 1
        assert summary.experiments["B"].failure_count == 1
        assert summary.experiments["B"].success_rate == 0.5

        assert summary.experiments["C"].success_count == 0
        assert summary.experiments["C"].failure_count == 2
        assert summary.experiments["C"].success_rate == 0.0

        assert summary.experiments["D"].success_count == 2
        assert summary.experiments["D"].failure_count == 0
        assert summary.experiments["D"].success_rate == 1.0

    def test_metrics_aggregated_correctly(self):
        """Numeric metrics are aggregated with correct mean/min/max."""
        task_hash = "e" * 64
        records = []

        # A: tokens 1000, 1200
        records.append(_build_raw_record("A", "task1", task_hash, 42, text_tokens=1000))
        records.append(_build_raw_record("A", "task1", task_hash, 43, text_tokens=1200))

        # B/C/D: same pattern
        for exp_name in ["B", "C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", task_hash, 42, text_tokens=1000))
            records.append(_build_raw_record(exp_name, "task1", task_hash, 43, text_tokens=1200))

        summary = summarize_raw_records(records)

        a_tokens = summary.experiments["A"].metrics["text_tokens"]
        assert a_tokens.count == 2
        assert a_tokens.mean == 1100.0
        assert a_tokens.min == 1000.0
        assert a_tokens.max == 1200.0

    def test_derived_formulas_correct(self):
        """Derived metrics use correct formulas."""
        task_hash = "f" * 64
        records = []

        # A: 1000 tokens, 5000 chars, 200ms
        records.append(_build_raw_record("A", "t", task_hash, 42, text_tokens=1000, text_chars=5000, latency_ms=200.0))

        # B: 750 tokens (25% saving), 4000 chars (20% saving), 180ms (10% improvement)
        records.append(_build_raw_record("B", "t", task_hash, 42, text_tokens=750, text_chars=4000, latency_ms=180.0))

        # C: same as B
        records.append(_build_raw_record("C", "t", task_hash, 42, text_tokens=750, text_chars=4000, latency_ms=180.0))

        # D: 600 tokens, 3200 chars, 150ms, memory 10/8 effective
        records.append(
            _build_raw_record(
                "D", "t", task_hash, 42,
                text_tokens=600,
                text_chars=3200,
                latency_ms=150.0,
                repeated_tool_calls=2,
                memory_used=10,
                memory_effective=8,
            )
        )

        summary = summarize_raw_records(records)

        # B vs A
        b_vs_a = summary.derived["B_vs_A"]
        assert b_vs_a.token_saving_rate == 0.25
        assert b_vs_a.char_saving_rate == 0.20
        assert b_vs_a.latency_improvement_rate == 0.10

        # D vs C: token saving = (750 - 600) / 750 = 0.2
        d_vs_c = summary.derived["D_vs_C"]
        assert d_vs_c.token_saving_rate == 0.2
        assert d_vs_c.effective_hit_rate == 0.8

    def test_task_count_correct(self):
        """task_count reflects unique task hashes."""
        records = []

        # 2 unique tasks × 4 experiments × 1 seed
        for exp_name in ["A", "B", "C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", "a" * 64, 42))
            records.append(_build_raw_record(exp_name, "task2", "b" * 64, 42))

        summary = summarize_raw_records(records)

        assert summary.task_count == 2

    def test_seed_count_correct(self):
        """seed_count reflects unique seeds."""
        task_hash = _task_hash("seed-count")
        records = []

        # 1 task × 4 experiments × 3 seeds
        for exp_name in ["A", "B", "C", "D"]:
            for seed in [42, 43, 44]:
                records.append(_build_raw_record(exp_name, "task1", task_hash, seed))

        summary = summarize_raw_records(records)

        assert summary.seed_count == 3


class TestFairnessValidation:
    """Test fairness validation between experiments."""

    def test_missing_experiment_rejected(self):
        """Missing B experiment is rejected."""
        task_hash = _task_hash("missing-experiment")
        records = []

        # Only A, C, D
        for exp_name in ["A", "C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", task_hash, 42))

        with pytest.raises(ValueError, match="missing: \\['B'\\]"):
            summarize_raw_records(records)

    def test_task_hash_mismatch_rejected(self):
        """Different task_hash between experiments is rejected."""
        records = []

        # A uses one task_hash
        records.append(_build_raw_record("A", "task1", _task_hash("task-a"), 42))

        # B/C/D use different task_hash
        for exp_name in ["B", "C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", _task_hash("task-b"), 42))

        with pytest.raises(ValueError, match="Fairness violation"):
            summarize_raw_records(records)

    def test_seed_mismatch_rejected(self):
        """Different seeds between experiments is rejected."""
        task_hash = _task_hash("seed-mismatch")
        records = []

        # A uses seeds 42, 43
        records.append(_build_raw_record("A", "task1", task_hash, 42))
        records.append(_build_raw_record("A", "task1", task_hash, 43))

        # B uses seeds 42, 44 (different)
        records.append(_build_raw_record("B", "task1", task_hash, 42))
        records.append(_build_raw_record("B", "task1", task_hash, 44))

        # C/D use 42, 43
        for exp_name in ["C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", task_hash, 42))
            records.append(_build_raw_record(exp_name, "task1", task_hash, 43))

        with pytest.raises(ValueError, match="Fairness violation"):
            summarize_raw_records(records)


class TestMetricsValidation:
    """Test metrics schema validation."""

    def test_invalid_metrics_schema_rejected(self):
        """Invalid metrics schema is rejected."""
        task_hash = _task_hash("invalid-metrics")
        record = _build_raw_record("A", "task1", task_hash, 42)

        # Corrupt metrics
        record.run_result.metrics["text_tokens"] = "not a number"

        records = [record]
        for exp_name in ["B", "C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", task_hash, 42))

        with pytest.raises(ValueError, match="Invalid metrics"):
            summarize_raw_records(records)

    def test_success_mismatch_rejected(self):
        """Metrics success != run_result.success is rejected."""
        task_hash = _task_hash("success-mismatch")
        record = _build_raw_record("A", "task1", task_hash, 42, success=True)

        # Corrupt: metrics says success=False but run_result says success=True
        record.run_result.metrics["success"] = False

        records = [record]
        for exp_name in ["B", "C", "D"]:
            records.append(_build_raw_record(exp_name, "task1", task_hash, 42))

        with pytest.raises(ValueError, match="Metrics success .* does not match"):
            summarize_raw_records(records)


class TestInputValidation:
    """Test input validation for summarize_raw_records."""

    def test_empty_records_rejected(self):
        """Empty records list is rejected."""
        with pytest.raises(ValueError, match="records must be a non-empty list"):
            summarize_raw_records([])

    def test_wrong_records_type_rejected(self):
        """Non-list records is rejected."""
        with pytest.raises(TypeError, match="records must be a list"):
            summarize_raw_records("not a list")  # type: ignore[arg-type]

    def test_wrong_record_item_type_rejected(self):
        """Non-RawRunRecord item is rejected."""
        with pytest.raises(TypeError, match="records\\[0\\] must be a RawRunRecord"):
            summarize_raw_records([{"not": "a RawRunRecord"}])  # type: ignore[list-item]


class TestExactMemoryTotals:
    """Test that memory totals are exact sums, not count * int(mean)."""

    def test_exact_memory_totals_in_derived_metrics(self):
        """Memory effective hit rate uses exact totals, not count * int(mean)."""
        task_hash = _task_hash("memory-totals")
        records = []

        # A/B/C with no memory (baseline experiments)
        for exp_name in ["A", "B", "C"]:
            records.append(_build_raw_record(exp_name, "task1", task_hash, 42))
            records.append(_build_raw_record(exp_name, "task1", task_hash, 43))

        # D with non-uniform memory usage across seeds
        # seed 42: memory_used=1, memory_effective=1
        records.append(
            _build_raw_record(
                "D",
                "task1",
                task_hash,
                42,
                memory_used=1,
                memory_effective=1,
            )
        )

        # seed 43: memory_used=2, memory_effective=1
        records.append(
            _build_raw_record(
                "D",
                "task1",
                task_hash,
                43,
                memory_used=2,
                memory_effective=1,
            )
        )

        summary = summarize_raw_records(records)

        # D experiment totals
        # memory_used: 1 + 2 = 3
        # memory_effective: 1 + 1 = 2
        # effective_hit_rate = 2 / 3

        # If using count * int(mean), would get:
        # mean(memory_used) = 1.5
        # count * int(1.5) = 2 * 1 = 2 (WRONG, should be 3)

        d_vs_c = summary.derived["D_vs_C"]
        assert d_vs_c.effective_hit_rate == pytest.approx(2 / 3)


class TestWriteSummary:
    """Test write_summary function."""

    def test_summary_written_as_utf8(self, tmp_path: Path):
        """Summary is written as UTF-8 JSON."""
        summary_path = tmp_path / "summary.json"

        summary = BenchmarkSummary(
            total_records=4,
            task_count=1,
            seed_count=1,
            experiments={
                "A": ExperimentSummary(
                    experiment=ExperimentName.A,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "B": ExperimentSummary(
                    experiment=ExperimentName.B,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "C": ExperimentSummary(
                    experiment=ExperimentName.C,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "D": ExperimentSummary(
                    experiment=ExperimentName.D,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
            },
            derived={
                "B_vs_A": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "C_vs_B": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "D_vs_C": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
            },
        )

        write_summary(summary_path, summary)

        # Verify file exists and is UTF-8
        assert summary_path.exists()
        content = summary_path.read_text(encoding="utf-8")
        parsed = json.loads(content)

        assert parsed["total_records"] == 4
        assert parsed["task_count"] == 1

    def test_summary_json_deterministic(self, tmp_path: Path):
        """Summary JSON is deterministic (sorted keys)."""
        summary_path = tmp_path / "summary.json"

        summary = BenchmarkSummary(
            total_records=4,
            task_count=1,
            seed_count=1,
            experiments={
                "A": ExperimentSummary(
                    experiment=ExperimentName.A,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "B": ExperimentSummary(
                    experiment=ExperimentName.B,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "C": ExperimentSummary(
                    experiment=ExperimentName.C,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "D": ExperimentSummary(
                    experiment=ExperimentName.D,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
            },
            derived={
                "B_vs_A": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "C_vs_B": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "D_vs_C": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
            },
        )

        write_summary(summary_path, summary)

        content = summary_path.read_text(encoding="utf-8")

        # Check that keys appear in sorted order
        assert content.index('"derived"') < content.index('"experiments"')
        assert content.index('"experiments"') < content.index('"seed_count"')

    def test_write_refuses_overwrite(self, tmp_path: Path):
        """write_summary refuses to overwrite existing file."""
        summary_path = tmp_path / "summary.json"

        summary = BenchmarkSummary(
            total_records=4,
            task_count=1,
            seed_count=1,
            experiments={
                "A": ExperimentSummary(
                    experiment=ExperimentName.A,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "B": ExperimentSummary(
                    experiment=ExperimentName.B,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "C": ExperimentSummary(
                    experiment=ExperimentName.C,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "D": ExperimentSummary(
                    experiment=ExperimentName.D,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
            },
            derived={
                "B_vs_A": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "C_vs_B": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "D_vs_C": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
            },
        )

        # First write succeeds
        write_summary(summary_path, summary)

        # Second write fails
        with pytest.raises(FileExistsError):
            write_summary(summary_path, summary)

    def test_wrong_path_type_rejected(self, tmp_path: Path):
        """Non-str/Path path is rejected."""
        summary = BenchmarkSummary(
            total_records=4,
            task_count=1,
            seed_count=1,
            experiments={
                "A": ExperimentSummary(
                    experiment=ExperimentName.A,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "B": ExperimentSummary(
                    experiment=ExperimentName.B,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "C": ExperimentSummary(
                    experiment=ExperimentName.C,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
                "D": ExperimentSummary(
                    experiment=ExperimentName.D,
                    run_count=1,
                    success_count=1,
                    failure_count=0,
                    success_rate=1.0,
                    metrics={},
                ),
            },
            derived={
                "B_vs_A": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "C_vs_B": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
                "D_vs_C": DerivedMetrics(
                    token_saving_rate=None,
                    char_saving_rate=None,
                    latency_improvement_rate=None,
                    repeat_work_reduction_rate=None,
                    effective_hit_rate=None,
                ),
            },
        )

        with pytest.raises(TypeError, match="path must be a str or Path"):
            write_summary(123, summary)  # type: ignore[arg-type]

    def test_wrong_summary_type_rejected(self, tmp_path: Path):
        """Non-BenchmarkSummary summary is rejected."""
        with pytest.raises(TypeError, match="summary must be a BenchmarkSummary"):
            write_summary(tmp_path / "summary.json", {"not": "a summary"})  # type: ignore[arg-type]


def _task_hash(label: str) -> str:
    """Generate a valid SHA-256 task hash from a label for test fixtures."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()
