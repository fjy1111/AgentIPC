"""Tests for A/B/C/D suite runner."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.experiment import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    EXPERIMENT_D,
    ExperimentConfig,
)
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.runner import RawRunRecord, run_abcd_suite
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.state.hub import StateHub


EXPECTED_ANSWER = "Network troubleshooting completed."
EMBEDDING_DIM = 64

KNOWLEDGE = [
    {
        "document_id": "network-diag",
        "text": "NetworkManager handles openEuler network diagnostics.",
        "keywords": ["NetworkManager", "openEuler", "network"],
    },
]


def _build_agents() -> AgentRegistry:
    """Build agent registry with all four required agents."""
    registry = AgentRegistry()
    registry.register(PlannerAgent())
    registry.register(RetrieverAgent(KNOWLEDGE))
    registry.register(ExecutorAgent())
    registry.register(SummarizerAgent())
    return registry


def _build_provider_bundle() -> ProviderBundle:
    """Build mock provider bundle."""
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={"network": EXPECTED_ANSWER},
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(dim=EMBEDDING_DIM),
    )


def _build_memory_service(tmp_path: Path, name: str) -> MemoryService:
    """Build memory service with fresh SQLite store."""
    store = SQLiteMemoryStore(tmp_path / f"memory_{name}.db")
    embedding = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    return MemoryService(store, embedding, VectorIndex(embedding.dim))


def _build_context(
    tmp_path: Path,
    *,
    experiment: ExperimentConfig,
    task_index: int,
    run_index: int,
    seed: int,
) -> RunContext:
    """Build a fresh RunContext for testing."""
    exp_name = experiment.name.value
    trace_path = tmp_path / f"trace_{exp_name}_{task_index}_{run_index}_{seed}.jsonl"
    return RunContext(
        trace_id=f"trace-{exp_name}-{task_index}-{run_index}-{seed}",
        task_id=f"task-{exp_name}-{task_index}-{run_index}-{seed}",
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(random_seed=seed),
        registry=CapabilityRegistry(),
        state_hub=StateHub(transport="inproc"),
        artifact_store=ArtifactStore(tmp_path / f"artifacts_{exp_name}_{task_index}_{run_index}"),
        memory_service=_build_memory_service(
            tmp_path, f"{exp_name}_{task_index}_{run_index}_{seed}"
        ),
        metrics=MetricsCollector(),
        trace_logger=TraceLogger(trace_path),
        provider_bundle=_build_provider_bundle(),
        use_state=False,
        use_memory=False,
        use_sandbox=False,
    )


class TestBasicSuiteExecution:
    """Test basic A/B/C/D suite execution."""

    def test_suite_produces_correct_number_of_records(self, tmp_path: Path):
        """4 configs × 2 tasks × 2 seeds = 16 records."""
        tasks = [
            "diagnose openEuler network connectivity",
            "inspect openEuler service status",
        ]
        seeds = [42, 43]

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        assert len(records) == 16

    def test_all_records_successful(self, tmp_path: Path):
        """All records complete successfully."""
        tasks = ["diagnose openEuler network connectivity"]
        seeds = [42]

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        for record in records:
            assert record.run_result.success is True


class TestConfigMajorOrder:
    """Test that suite executes in config-major order (A→B→C→D)."""

    def test_config_order_is_abcd(self, tmp_path: Path):
        """Records are grouped by experiment: A×4, B×4, C×4, D×4."""
        tasks = ["task1", "task2"]
        seeds = [42, 43]

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # Extract experiment names
        experiment_names = [record.experiment.name.value for record in records]

        # Should be: A×4, B×4, C×4, D×4
        expected = ["A"] * 4 + ["B"] * 4 + ["C"] * 4 + ["D"] * 4
        assert experiment_names == expected


class TestTaskOrder:
    """Test that task order is preserved within each config block."""

    def test_task_order_preserved(self, tmp_path: Path):
        """Within each config, tasks appear in input order."""
        tasks = ["task_first", "task_second"]
        seeds = [42, 43]

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # Check task order within first config (A)
        first_config_records = records[0:4]
        first_config_tasks = [r.task for r in first_config_records]
        # Should be: task_first (2 seeds), task_second (2 seeds)
        assert first_config_tasks == [
            "task_first",
            "task_first",
            "task_second",
            "task_second",
        ]


class TestSeedOrder:
    """Test that seed order is preserved."""

    def test_seed_order_preserved(self, tmp_path: Path):
        """Seeds appear in input order, not sorted."""
        tasks = ["task1"]
        seeds = [44, 42]  # Deliberately unsorted

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # Check seeds in first config (A)
        first_config_seeds = [records[0].seed, records[1].seed]
        assert first_config_seeds == [44, 42]


class TestFactoryCallParameters:
    """Test that factory receives correct parameters."""

    def test_factory_call_sequence(self, tmp_path: Path):
        """Factory is called with correct (experiment, task_index, run_index, seed, task)."""
        tasks = ["task0", "task1"]
        seeds = [42, 43]
        calls: list[tuple[str, int, int, int, str]] = []

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            calls.append((experiment.name.value, task_index, run_index, seed, task))
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # Expected call sequence: A (all tasks/seeds), then B, C, D
        expected = [
            # A
            ("A", 0, 0, 42, "task0"),
            ("A", 0, 1, 43, "task0"),
            ("A", 1, 0, 42, "task1"),
            ("A", 1, 1, 43, "task1"),
            # B
            ("B", 0, 0, 42, "task0"),
            ("B", 0, 1, 43, "task0"),
            ("B", 1, 0, 42, "task1"),
            ("B", 1, 1, 43, "task1"),
            # C
            ("C", 0, 0, 42, "task0"),
            ("C", 0, 1, 43, "task0"),
            ("C", 1, 0, 42, "task1"),
            ("C", 1, 1, 43, "task1"),
            # D
            ("D", 0, 0, 42, "task0"),
            ("D", 0, 1, 43, "task0"),
            ("D", 1, 0, 42, "task1"),
            ("D", 1, 1, 43, "task1"),
        ]

        assert calls == expected


class TestTaskHashFairness:
    """Test that A/B/C/D use identical task inputs (core acceptance)."""

    def test_task_hash_identical_across_configs(self, tmp_path: Path):
        """For each task_index/run_index, A/B/C/D have identical task_hash."""
        tasks = ["task0", "task1"]
        seeds = [42, 43]

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # Check each task_index/run_index combination
        for task_index in range(2):
            for run_index in range(2):
                # Extract A/B/C/D records for this combination
                indices = [
                    0 * 4 + task_index * 2 + run_index,  # A
                    1 * 4 + task_index * 2 + run_index,  # B
                    2 * 4 + task_index * 2 + run_index,  # C
                    3 * 4 + task_index * 2 + run_index,  # D
                ]
                config_records = [records[i] for i in indices]

                # All task_hashes must be identical
                task_hashes = [r.task_hash for r in config_records]
                assert len(set(task_hashes)) == 1

                # All tasks must be identical
                tasks_used = [r.task for r in config_records]
                assert len(set(tasks_used)) == 1

                # All seeds must be identical
                seeds_used = [r.seed for r in config_records]
                assert len(set(seeds_used)) == 1

    def test_unicode_task_hash_fairness(self, tmp_path: Path):
        """Unicode tasks also have consistent hash across configs."""
        tasks = ["diagnose openEuler network connectivity", "诊断 openEuler 网络连接"]
        seeds = [42]

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # Check Unicode task (task_index=1, run_index=0)
        unicode_records = [
            records[0 * 2 + 1 * 1 + 0],  # A
            records[1 * 2 + 1 * 1 + 0],  # B
            records[2 * 2 + 1 * 1 + 0],  # C
            records[3 * 2 + 1 * 1 + 0],  # D
        ]

        task_hashes = [r.task_hash for r in unicode_records]
        assert len(set(task_hashes)) == 1

        # Verify it matches expected hash
        expected_hash = hashlib.sha256("诊断 openEuler 网络连接".encode("utf-8")).hexdigest()
        assert task_hashes[0] == expected_hash


class TestDuplicateTasks:
    """Test that duplicate tasks are preserved."""

    def test_duplicate_tasks_preserved(self, tmp_path: Path):
        """Duplicate tasks produce separate task_index records."""
        tasks = ["same task", "same task"]
        seeds = [42]
        task_indices_received: list[int] = []

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            task_indices_received.append(task_index)
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # Should produce 4 configs × 2 tasks × 1 seed = 8 records
        assert len(records) == 8

        # Factory should receive task_index 0 and 1 for each config
        assert 0 in task_indices_received
        assert 1 in task_indices_received


class TestInvalidTasksRejection:
    """Test that invalid tasks are rejected before execution."""

    @pytest.mark.parametrize(
        "invalid_tasks,expected_error",
        [
            (None, TypeError),
            ((), TypeError),
            ("task", TypeError),
            ([], ValueError),
            ([""], ValueError),
            (["valid", ""], ValueError),
            ([123], TypeError),
            (["valid", None], TypeError),
        ],
        ids=[
            "None",
            "tuple",
            "str",
            "empty_list",
            "empty_string",
            "valid_then_empty",
            "int",
            "valid_then_None",
        ],
    )
    def test_invalid_tasks_rejected_before_factory_call(
        self,
        tmp_path: Path,
        invalid_tasks: Any,
        expected_error: type[Exception],
    ):
        """Invalid tasks raise error with zero factory calls."""
        call_count = 0

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            nonlocal call_count
            call_count += 1
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        with pytest.raises(expected_error):
            run_abcd_suite(
                tasks=invalid_tasks,
                seeds=[42],
                run_factory=factory,
            )

        assert call_count == 0


class TestInvalidSeedsRejection:
    """Test that invalid seeds are rejected before execution."""

    @pytest.mark.parametrize(
        "invalid_seeds,expected_error",
        [
            ([], ValueError),
            ([True], TypeError),
            ([-1], ValueError),
            ([42, True], TypeError),
            ([42, -1], ValueError),
        ],
        ids=["empty", "bool", "negative", "valid_then_bool", "valid_then_negative"],
    )
    def test_invalid_seeds_rejected_before_factory_call(
        self,
        tmp_path: Path,
        invalid_seeds: Any,
        expected_error: type[Exception],
    ):
        """Invalid seeds raise error with zero factory calls."""
        call_count = 0

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            nonlocal call_count
            call_count += 1
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        with pytest.raises(expected_error):
            run_abcd_suite(
                tasks=["task"],
                seeds=invalid_seeds,
                run_factory=factory,
            )

        assert call_count == 0


class TestFailureRecordContinuation:
    """Test that failed runs don't stop the suite."""

    def test_failure_in_one_config_continues_suite(self, tmp_path: Path):
        """Failure in B doesn't stop C and D."""
        tasks = ["task1"]
        seeds = [42, 43]

        class FailingPlannerAgent(PlannerAgent):
            """Planner that fails during handle."""

            def handle(
                self,
                envelope: AgentEnvelope,
                ctx: RunContext,
            ) -> AgentEnvelope:
                raise RuntimeError("suite-boom")

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            ctx = _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed)

            # Make B with task_index=0, seed=43 fail
            if experiment.name.value == "B" and task_index == 0 and seed == 43:
                registry = AgentRegistry()
                registry.register(FailingPlannerAgent())
                registry.register(RetrieverAgent(KNOWLEDGE))
                registry.register(ExecutorAgent())
                registry.register(SummarizerAgent())
            else:
                registry = _build_agents()

            return ctx, registry

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # All 8 records should be present (4 configs × 1 task × 2 seeds)
        assert len(records) == 8

        # Find the failing record (B, task_index=0, run_index=1)
        b_records = records[2:4]  # B block
        assert b_records[0].run_result.success is True  # seed 42
        assert b_records[1].run_result.success is False  # seed 43
        assert b_records[1].run_result.error is not None
        assert b_records[1].run_result.error["type"] == "RuntimeError"
        assert b_records[1].run_result.error["message"] == "suite-boom"

        # C and D should still execute successfully
        c_records = records[4:6]
        d_records = records[6:8]
        for record in c_records + d_records:
            assert record.run_result.success is True


class TestMetricsIndependence:
    """Test that metrics don't accumulate across runs."""

    def test_metrics_fresh_per_run(self, tmp_path: Path):
        """Each record has exactly 8 messages, not accumulating."""
        tasks = ["task1", "task2"]
        seeds = [42, 43]

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        # All records should have exactly 8 messages
        for record in records:
            assert record.run_result.metrics["message_count"] == 8


class TestJSONCompatibility:
    """Test that all records can be serialized to JSON."""

    def test_all_records_json_serializable(self, tmp_path: Path):
        """All records can be serialized."""
        tasks = ["task1"]
        seeds = [42]

        def factory(
            experiment: ExperimentConfig,
            task_index: int,
            run_index: int,
            seed: int,
            task: str,
        ) -> tuple[RunContext, AgentRegistry]:
            return (
                _build_context(tmp_path, experiment=experiment, task_index=task_index, run_index=run_index, seed=seed),
                _build_agents(),
            )

        records = run_abcd_suite(
            tasks=tasks,
            seeds=seeds,
            run_factory=factory,
        )

        for record in records:
            data = record.model_dump(mode="json")
            assert isinstance(data, dict)


class TestInputValidation:
    """Test input validation for run_abcd_suite."""

    def test_non_callable_factory_rejected(self, tmp_path: Path):
        """Non-callable factory raises TypeError."""
        with pytest.raises(TypeError, match="run_factory must be callable"):
            run_abcd_suite(
                tasks=["task"],
                seeds=[42],
                run_factory="not_callable",  # type: ignore[arg-type]
            )
