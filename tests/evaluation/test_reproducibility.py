"""Tests for benchmark reproducibility across independent runs.

This module verifies that the A/B/C/D benchmark produces deterministic results
when run with fixed seeds and mock providers. It exercises the complete pipeline:
raw records → summary → report, and validates that two independent suites
produce identical non-timing fields.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from itertools import count
from pathlib import Path
from uuid import UUID

import agentipc.utils

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.env import capture_environment_snapshot
from agentipc.evaluation.experiment import ExperimentConfig
from agentipc.evaluation.io import (
    append_raw_record,
    create_result_dir,
    read_raw_records,
    summarize_raw_records,
    write_summary,
)
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.report import render_markdown_report, write_markdown_report
from agentipc.evaluation.runner import RawRunRecord, run_abcd_suite
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.state.hub import StateHub


EMBEDDING_DIM = 64
FIXED_TASK = "diagnose openEuler network connectivity"
FIXED_SEEDS = [42, 43]
EXPECTED_ANSWER = "Network troubleshooting completed."

KNOWLEDGE = [
    {
        "document_id": "network-diag",
        "text": "NetworkManager handles openEuler network diagnostics.",
        "keywords": ["NetworkManager", "openEuler", "network"],
    },
]


class _FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        fixed = cls(
            2026,
            1,
            1,
            0,
            0,
            0,
            123456,
            tzinfo=timezone.utc,
        )
        if tz is None:
            return fixed
        return fixed.astimezone(tz)


def _reset_protocol_randomness(monkeypatch) -> None:
    # Protocol communication metrics include AgentEnvelope.message_id and
    # AgentEnvelope.created_at. T131 freezes these volatile metadata sources
    # so independent suites exercise identical deterministic inputs.
    sequence = count(1)

    def deterministic_uuid4():
        return UUID(int=next(sequence))

    monkeypatch.setattr(agentipc.utils, "uuid4", deterministic_uuid4)
    monkeypatch.setattr(agentipc.utils, "datetime", _FixedDateTime)


def _build_deterministic_context(
    tmp_root: Path,
    suite_id: str,
    *,
    experiment: ExperimentConfig,
    task_index: int,
    run_index: int,
    seed: int,
) -> RunContext:
    """Build a RunContext with deterministic IDs for reproducibility testing.

    Logical IDs (trace_id, task_id) are identical across independent suites
    for the same logical run (experiment, task_index, run_index, seed).

    suite_id is used only for physical isolation of storage paths
    (trace files, artifacts, memory databases).
    """
    exp_name = experiment.name.value

    # Logical IDs: identical across suites for same logical run
    trace_id = f"trace-{exp_name}-{task_index}-{run_index}-{seed}"
    task_id = f"task-{exp_name}-{task_index}-{run_index}-{seed}"

    # Physical paths: suite_id isolates storage
    trace_path = tmp_root / f"trace_{suite_id}_{exp_name}_{task_index}_{run_index}_{seed}.jsonl"

    return RunContext(
        trace_id=trace_id,
        task_id=task_id,
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(random_seed=seed),
        registry=CapabilityRegistry(),
        state_hub=StateHub(transport="inproc"),
        artifact_store=ArtifactStore(
            tmp_root / f"artifacts_{suite_id}_{exp_name}_{task_index}_{run_index}"
        ),
        memory_service=_build_memory_service(
            tmp_root, f"{suite_id}_{exp_name}_{task_index}_{run_index}_{seed}"
        ),
        metrics=MetricsCollector(),
        trace_logger=TraceLogger(trace_path),
        provider_bundle=_build_provider_bundle(),
        use_state=False,
        use_memory=False,
        use_sandbox=False,
    )


def _build_memory_service(tmp_path: Path, name: str) -> MemoryService:
    """Build fresh memory service with SQLite store and hash embedding."""
    store = SQLiteMemoryStore(tmp_path / f"memory_{name}.db")
    embedding = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    return MemoryService(store, embedding, VectorIndex(embedding.dim))


def _build_provider_bundle() -> ProviderBundle:
    """Build mock provider bundle with keyword-based responses."""
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={"network": EXPECTED_ANSWER},
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(dim=EMBEDDING_DIM),
    )


def _build_agents() -> AgentRegistry:
    """Build agent registry with all four required agents."""
    registry = AgentRegistry()
    registry.register(PlannerAgent())
    registry.register(RetrieverAgent(KNOWLEDGE))
    registry.register(ExecutorAgent())
    registry.register(SummarizerAgent())
    return registry


class TestTwoIndependentSuites:
    """Test that two independent Mock suites produce reproducible results."""

    def test_two_suites_produce_identical_deterministic_fields(
        self, tmp_path: Path, monkeypatch
    ):
        """Two independent suites with same seeds produce identical non-timing fields."""
        # Suite 1
        suite1_root = tmp_path / "suite1"
        suite1_root.mkdir()

        def factory1(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_deterministic_context(
                    suite1_root,
                    "suite1",
                    experiment=experiment,
                    task_index=task_index,
                    run_index=run_index,
                    seed=seed,
                ),
                _build_agents(),
            )

        _reset_protocol_randomness(monkeypatch)
        records1 = run_abcd_suite(
            tasks=[FIXED_TASK],
            seeds=FIXED_SEEDS,
            run_factory=factory1,
        )

        # Suite 2
        suite2_root = tmp_path / "suite2"
        suite2_root.mkdir()

        def factory2(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_deterministic_context(
                    suite2_root,
                    "suite2",
                    experiment=experiment,
                    task_index=task_index,
                    run_index=run_index,
                    seed=seed,
                ),
                _build_agents(),
            )

        _reset_protocol_randomness(monkeypatch)
        records2 = run_abcd_suite(
            tasks=[FIXED_TASK],
            seeds=FIXED_SEEDS,
            run_factory=factory2,
        )

        # Same length
        assert len(records1) == len(records2) == 8

        # Compare each pair of records
        for rec1, rec2 in zip(records1, records2):
            # Experiment config must match
            assert rec1.experiment == rec2.experiment

            # Task and task_hash must match
            assert rec1.task == rec2.task
            assert rec1.task_hash == rec2.task_hash

            # Seed must match
            assert rec1.seed == rec2.seed

            # Sandbox flag must match
            assert rec1.use_sandbox == rec2.use_sandbox

            # RunResult deterministic fields must match
            assert rec1.run_result.task_id == rec2.run_result.task_id
            assert rec1.run_result.success == rec2.run_result.success
            assert rec1.run_result.answer == rec2.run_result.answer
            assert rec1.run_result.error == rec2.run_result.error

            # Metrics: all fields except latency_ms and llm_latency_ms
            m1 = rec1.run_result.metrics
            m2 = rec2.run_result.metrics

            # Deterministic metric fields
            deterministic_fields = [
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
                "llm_call_count",
                "llm_prompt_tokens",
                "llm_completion_tokens",
                "llm_total_tokens",
                "llm_usage_missing_count",
                "llm_latency_ms",  # MockLLMProvider returns 0, so deterministic
                "success",
            ]

            for field in deterministic_fields:
                assert m1[field] == m2[field], f"Mismatch in {field}: {m1[field]} != {m2[field]}"

            # latency_ms is real-time measurement, exclude from comparison


class TestSummaryReproducibility:
    """Test that summaries from two suites have identical non-timing fields."""

    def test_summaries_match_except_timing(self, tmp_path: Path, monkeypatch):
        """Summaries from two suites match except for latency metrics."""
        # Suite 1
        suite1_root = tmp_path / "suite1"
        suite1_root.mkdir()

        def factory1(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_deterministic_context(
                    suite1_root,
                    "suite1",
                    experiment=experiment,
                    task_index=task_index,
                    run_index=run_index,
                    seed=seed,
                ),
                _build_agents(),
            )

        _reset_protocol_randomness(monkeypatch)
        records1 = run_abcd_suite(
            tasks=[FIXED_TASK],
            seeds=FIXED_SEEDS,
            run_factory=factory1,
        )

        summary1 = summarize_raw_records(records1)

        # Suite 2
        suite2_root = tmp_path / "suite2"
        suite2_root.mkdir()

        def factory2(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_deterministic_context(
                    suite2_root,
                    "suite2",
                    experiment=experiment,
                    task_index=task_index,
                    run_index=run_index,
                    seed=seed,
                ),
                _build_agents(),
            )

        _reset_protocol_randomness(monkeypatch)
        records2 = run_abcd_suite(
            tasks=[FIXED_TASK],
            seeds=FIXED_SEEDS,
            run_factory=factory2,
        )

        summary2 = summarize_raw_records(records2)

        # Top-level fields must match
        assert summary1.total_records == summary2.total_records
        assert summary1.task_count == summary2.task_count
        assert summary1.seed_count == summary2.seed_count

        # Experiment summaries must match (except latency)
        for exp_name in ["A", "B", "C", "D"]:
            exp1 = summary1.experiments[exp_name]
            exp2 = summary2.experiments[exp_name]

            assert exp1.experiment == exp2.experiment
            assert exp1.run_count == exp2.run_count
            assert exp1.success_count == exp2.success_count
            assert exp1.failure_count == exp2.failure_count
            assert exp1.success_rate == exp2.success_rate

            # Compare all metrics except latency_ms
            for metric_name, stats1 in exp1.metrics.items():
                if metric_name == "latency_ms":
                    continue  # Skip timing metric

                stats2 = exp2.metrics[metric_name]
                assert stats1.count == stats2.count, (
                    f"{exp_name}.{metric_name}.count: {stats1.count} != {stats2.count}"
                )
                assert stats1.mean == stats2.mean, (
                    f"{exp_name}.{metric_name}.mean: {stats1.mean} != {stats2.mean}"
                )
                assert stats1.std == stats2.std, (
                    f"{exp_name}.{metric_name}.std: {stats1.std} != {stats2.std}"
                )
                assert stats1.min == stats2.min, (
                    f"{exp_name}.{metric_name}.min: {stats1.min} != {stats2.min}"
                )
                assert stats1.max == stats2.max, (
                    f"{exp_name}.{metric_name}.max: {stats1.max} != {stats2.max}"
                )

        # Derived metrics must match (except latency_improvement_rate)
        for derived_key in ["B_vs_A", "C_vs_B", "D_vs_C"]:
            derived1 = summary1.derived[derived_key]
            derived2 = summary2.derived[derived_key]

            assert derived1.token_saving_rate == derived2.token_saving_rate
            assert derived1.char_saving_rate == derived2.char_saving_rate
            # Skip latency_improvement_rate (depends on latency_ms)
            assert derived1.repeat_work_reduction_rate == derived2.repeat_work_reduction_rate
            assert derived1.effective_hit_rate == derived2.effective_hit_rate


class TestEnvironmentSnapshotReproducibility:
    """Test that environment snapshots are reproducible."""

    def test_same_process_produces_identical_snapshots(self):
        """Two snapshots in same process are identical."""
        config = AgentIPCConfig()

        snapshot1 = capture_environment_snapshot(config)
        snapshot2 = capture_environment_snapshot(config)

        assert snapshot1 == snapshot2


class TestResultArtifactPipeline:
    """Test complete pipeline: raw → summary → report."""

    def test_pipeline_produces_all_artifacts(self, tmp_path: Path):
        """Complete pipeline produces raw.jsonl, summary.json, report.md."""
        # Run one small suite
        suite_root = tmp_path / "suite"
        suite_root.mkdir()

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_deterministic_context(
                    suite_root,
                    "test",
                    experiment=experiment,
                    task_index=task_index,
                    run_index=run_index,
                    seed=seed,
                ),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=[FIXED_TASK],
            seeds=FIXED_SEEDS,
            run_factory=factory,
        )

        # Create result directory
        result_dir = create_result_dir(tmp_path, "test-suite")

        # Write raw records
        raw_path = result_dir / "raw.jsonl"
        for record in records:
            append_raw_record(raw_path, record)

        # Verify raw.jsonl exists and can be read
        assert raw_path.exists()
        read_records = read_raw_records(raw_path)
        assert len(read_records) == len(records)

        # Generate summary
        summary = summarize_raw_records(records)
        summary_path = result_dir / "summary.json"
        write_summary(summary_path, summary)

        # Verify summary.json exists
        assert summary_path.exists()

        # Generate report
        report_path = result_dir / "report.md"
        write_markdown_report(report_path, summary)

        # Verify report.md exists
        assert report_path.exists()

        # Verify report contains expected sections
        report_content = report_path.read_text(encoding="utf-8")
        assert "# AgentIPC Benchmark Report" in report_content
        assert "## Communication" in report_content
        assert "## Derived Comparisons" in report_content


class TestRawRecordRoundTrip:
    """Test that raw records survive file round-trip."""

    def test_raw_record_file_round_trip_preserves_deterministic_fields(self, tmp_path: Path):
        """Writing and reading raw records preserves deterministic fields."""
        # Run small suite
        suite_root = tmp_path / "suite"
        suite_root.mkdir()

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_deterministic_context(
                    suite_root,
                    "test",
                    experiment=experiment,
                    task_index=task_index,
                    run_index=run_index,
                    seed=seed,
                ),
                _build_agents(),
            )

        original_records = run_abcd_suite(
            tasks=[FIXED_TASK],
            seeds=[42],  # Single seed for speed
            run_factory=factory,
        )

        # Write to file
        raw_path = tmp_path / "raw.jsonl"
        for record in original_records:
            append_raw_record(raw_path, record)

        # Read back
        restored_records = read_raw_records(raw_path)

        # Should have same count
        assert len(restored_records) == len(original_records)

        # Each record's deterministic fields should match
        for orig, restored in zip(original_records, restored_records):
            assert orig.experiment == restored.experiment
            assert orig.task == restored.task
            assert orig.task_hash == restored.task_hash
            assert orig.seed == restored.seed
            assert orig.use_sandbox == restored.use_sandbox
            assert orig.run_result.task_id == restored.run_result.task_id
            assert orig.run_result.success == restored.run_result.success
            assert orig.run_result.answer == restored.run_result.answer

            # Metrics deterministic fields
            orig_metrics = orig.run_result.metrics
            restored_metrics = restored.run_result.metrics

            for field in [
                "message_count",
                "text_chars",
                "text_tokens",
                "success",
            ]:
                assert orig_metrics[field] == restored_metrics[field]


class TestReportReproducibility:
    """Test that reports from two suites have same structure."""

    def test_two_reports_have_same_structure(self, tmp_path: Path):
        """Two reports from independent suites have same section structure."""
        # Suite 1
        suite1_root = tmp_path / "suite1"
        suite1_root.mkdir()

        def factory1(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_deterministic_context(
                    suite1_root,
                    "suite1",
                    experiment=experiment,
                    task_index=task_index,
                    run_index=run_index,
                    seed=seed,
                ),
                _build_agents(),
            )

        records1 = run_abcd_suite(
            tasks=[FIXED_TASK],
            seeds=[42],
            run_factory=factory1,
        )

        summary1 = summarize_raw_records(records1)
        report1 = render_markdown_report(summary1)

        # Suite 2
        suite2_root = tmp_path / "suite2"
        suite2_root.mkdir()

        def factory2(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_deterministic_context(
                    suite2_root,
                    "suite2",
                    experiment=experiment,
                    task_index=task_index,
                    run_index=run_index,
                    seed=seed,
                ),
                _build_agents(),
            )

        records2 = run_abcd_suite(
            tasks=[FIXED_TASK],
            seeds=[42],
            run_factory=factory2,
        )

        summary2 = summarize_raw_records(records2)
        report2 = render_markdown_report(summary2)

        # Both reports should have same sections
        required_sections = [
            "## Overview",
            "## Communication",
            "## State",
            "## Memory",
            "## LLM Usage",
            "## Latency",
            "## Derived Comparisons",
            "## Interpretation Notes",
        ]

        for section in required_sections:
            assert section in report1
            assert section in report2

        # Both should have A/B/C/D rows
        for exp in ["| A |", "| B |", "| C |", "| D |"]:
            assert exp in report1
            assert exp in report2

        # Both should have derived comparison rows
        for comp in ["| B vs A |", "| C vs B |", "| D vs C |"]:
            assert comp in report1
            assert comp in report2
