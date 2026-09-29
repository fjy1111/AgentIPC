"""Tests for repeated-run loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.experiment import EXPERIMENT_B, ExperimentConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.runner import RawRunRecord, run_repeated
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


TASK = "diagnose openEuler network connectivity"
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
    run_index: int,
    seed: int,
) -> RunContext:
    """Build a fresh RunContext for testing."""
    trace_path = tmp_path / f"trace_{run_index}_{seed}.jsonl"
    return RunContext(
        trace_id=f"trace-{run_index}-{seed}",
        task_id=f"task-{run_index}-{seed}",
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(random_seed=seed),
        registry=CapabilityRegistry(),
        state_hub=StateHub(transport="inproc"),
        artifact_store=ArtifactStore(tmp_path / f"artifacts_{run_index}"),
        memory_service=_build_memory_service(tmp_path, f"{run_index}_{seed}"),
        metrics=MetricsCollector(),
        trace_logger=TraceLogger(trace_path),
        provider_bundle=_build_provider_bundle(),
        use_state=False,
        use_memory=False,
        use_sandbox=False,
    )


class TestBasicRepeatedRun:
    """Test basic repeated run execution."""

    def test_repeated_run_produces_correct_number_of_records(self, tmp_path: Path):
        """Three seeds produce three records."""
        seeds = [42, 43, 44]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        assert len(records) == 3

    def test_repeated_run_preserves_seed_order(self, tmp_path: Path):
        """Seeds appear in output in same order as input."""
        seeds = [42, 43, 44]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        assert [record.seed for record in records] == [42, 43, 44]

    def test_repeated_run_all_successful(self, tmp_path: Path):
        """All runs complete successfully."""
        seeds = [42, 43, 44]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        for record in records:
            assert record.experiment == EXPERIMENT_B
            assert record.task == TASK
            assert record.run_result.success is True
            assert record.run_result.answer == EXPECTED_ANSWER

    def test_metrics_are_fresh_per_run(self, tmp_path: Path):
        """Each run has independent metrics (no accumulation)."""
        seeds = [42, 43, 44]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        # Each run should have exactly 8 messages, not 8/16/24
        for record in records:
            assert record.run_result.metrics["message_count"] == 8


class TestFactoryCallParameters:
    """Test that factory receives correct parameters."""

    def test_factory_receives_correct_run_index_and_seed(self, tmp_path: Path):
        """Factory is called with (0,42), (1,43), (2,44)."""
        seeds = [42, 43, 44]
        calls: list[tuple[int, int]] = []

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            calls.append((run_index, seed))
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        assert calls == [(0, 42), (1, 43), (2, 44)]


class TestTraceIsolation:
    """Test that each run produces independent trace."""

    def test_each_run_has_unique_trace_path(self, tmp_path: Path):
        """Each record has different trace_path."""
        seeds = [42, 43, 44]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        trace_paths = [record.run_result.trace_path for record in records]
        assert len(set(trace_paths)) == 3  # All unique

    def test_all_trace_files_exist(self, tmp_path: Path):
        """Each trace file is created."""
        seeds = [42, 43, 44]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        for record in records:
            assert Path(record.run_result.trace_path).exists()


class TestSeedOrderPreservation:
    """Test that seed order is never sorted or reordered."""

    def test_unordered_seeds_preserved(self, tmp_path: Path):
        """Seeds [44, 42, 43] produce records in same order."""
        seeds = [44, 42, 43]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        assert [record.seed for record in records] == [44, 42, 43]


class TestDuplicateSeeds:
    """Test that duplicate seeds are preserved."""

    def test_duplicate_seeds_produce_multiple_records(self, tmp_path: Path):
        """Seeds [42, 42, 43] produce three records."""
        seeds = [42, 42, 43]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        assert len(records) == 3
        assert [record.seed for record in records] == [42, 42, 43]

    def test_duplicate_seeds_call_factory_for_each(self, tmp_path: Path):
        """Factory is called (0,42), (1,42), (2,43)."""
        seeds = [42, 42, 43]
        calls: list[tuple[int, int]] = []

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            calls.append((run_index, seed))
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        assert calls == [(0, 42), (1, 42), (2, 43)]


class TestInvalidSeedsRejection:
    """Test that invalid seeds are rejected before any execution."""

    @pytest.mark.parametrize(
        "invalid_seeds,expected_error",
        [
            (None, TypeError),
            ((), TypeError),
            ("42", TypeError),
            ([True], TypeError),
            ([False], TypeError),
            (["42"], TypeError),
            ([42.0], TypeError),
            ([-1], ValueError),
            ([42, -1], ValueError),
            ([42, True], TypeError),
        ],
        ids=[
            "None",
            "tuple",
            "str",
            "list[True]",
            "list[False]",
            "list[str]",
            "list[float]",
            "negative",
            "valid_then_negative",
            "valid_then_bool",
        ],
    )
    def test_invalid_seeds_rejected_before_factory_call(
        self,
        tmp_path: Path,
        invalid_seeds: Any,
        expected_error: type[Exception],
    ):
        """Invalid seeds raise error with zero factory calls."""
        call_count = 0

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            nonlocal call_count
            call_count += 1
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        with pytest.raises(expected_error):
            run_repeated(
                experiment=EXPERIMENT_B,
                task=TASK,
                seeds=invalid_seeds,
                run_factory=factory,
            )

        assert call_count == 0


class TestEmptySeeds:
    """Test that empty seed list is rejected."""

    def test_empty_list_rejected(self, tmp_path: Path):
        """Empty list raises ValueError."""
        call_count = 0

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            nonlocal call_count
            call_count += 1
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        with pytest.raises(ValueError, match="seeds must be a non-empty list"):
            run_repeated(
                experiment=EXPERIMENT_B,
                task=TASK,
                seeds=[],
                run_factory=factory,
            )

        assert call_count == 0


class TestFactorySeedMismatch:
    """Test that factory returning wrong seed is detected."""

    def test_factory_wrong_seed_rejected(self, tmp_path: Path):
        """Factory returning ctx with wrong seed raises ValueError."""
        seeds = [43]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            # Factory receives seed=43 but returns ctx with seed=42
            ctx = _build_context(tmp_path, run_index=run_index, seed=42)
            return ctx, _build_agents()

        with pytest.raises(ValueError, match="random_seed=42.*expected 43"):
            run_repeated(
                experiment=EXPERIMENT_B,
                task=TASK,
                seeds=seeds,
                run_factory=factory,
            )

        # Verify no trace was created (business logic didn't run)
        trace_files = list(tmp_path.glob("*.jsonl"))
        assert len(trace_files) == 0


class TestFreshContextRequirement:
    """Test that factory creates fresh objects for each run."""

    def test_factory_creates_fresh_metrics(self, tmp_path: Path):
        """Each run gets a different MetricsCollector instance."""
        seeds = [42, 43, 44]
        metrics_objects: list[MetricsCollector] = []

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            ctx = _build_context(tmp_path, run_index=run_index, seed=seed)
            metrics_objects.append(ctx.metrics)
            return ctx, _build_agents()

        run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        # All metrics instances must be different
        assert len(metrics_objects) == 3
        assert metrics_objects[0] is not metrics_objects[1]
        assert metrics_objects[0] is not metrics_objects[2]
        assert metrics_objects[1] is not metrics_objects[2]

    def test_factory_creates_fresh_registry(self, tmp_path: Path):
        """Each run gets a different CapabilityRegistry instance."""
        seeds = [42, 43, 44]
        registry_objects: list[CapabilityRegistry] = []

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            ctx = _build_context(tmp_path, run_index=run_index, seed=seed)
            registry_objects.append(ctx.registry)
            return ctx, _build_agents()

        run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        # All registries must be different
        assert len(registry_objects) == 3
        assert registry_objects[0] is not registry_objects[1]
        assert registry_objects[0] is not registry_objects[2]
        assert registry_objects[1] is not registry_objects[2]

    def test_factory_creates_fresh_trace_logger(self, tmp_path: Path):
        """Each run gets a different TraceLogger instance."""
        seeds = [42, 43, 44]
        logger_objects: list[TraceLogger] = []

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            ctx = _build_context(tmp_path, run_index=run_index, seed=seed)
            logger_objects.append(ctx.trace_logger)
            return ctx, _build_agents()

        run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        # All loggers must be different
        assert len(logger_objects) == 3
        assert logger_objects[0] is not logger_objects[1]
        assert logger_objects[0] is not logger_objects[2]
        assert logger_objects[1] is not logger_objects[2]


class TestFailureRecordContinuation:
    """Test that failed runs don't stop later runs."""

    def test_failure_in_middle_continues_execution(self, tmp_path: Path):
        """Run 1 success, run 2 failure, run 3 success all complete."""
        seeds = [42, 43, 44]

        class FailingPlannerAgent(PlannerAgent):
            """Planner that fails during handle."""

            def handle(
                self,
                envelope: AgentEnvelope,
                ctx: RunContext,
            ) -> AgentEnvelope:
                raise RuntimeError("repeat-boom")

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            ctx = _build_context(tmp_path, run_index=run_index, seed=seed)

            # Make run_index=1 fail
            if run_index == 1:
                registry = AgentRegistry()
                registry.register(FailingPlannerAgent())
                registry.register(RetrieverAgent(KNOWLEDGE))
                registry.register(ExecutorAgent())
                registry.register(SummarizerAgent())
            else:
                registry = _build_agents()

            return ctx, registry

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        # All three records returned
        assert len(records) == 3

        # Check success pattern
        assert records[0].run_result.success is True
        assert records[1].run_result.success is False
        assert records[2].run_result.success is True

        # Check failure details
        assert records[1].run_result.error is not None
        assert records[1].run_result.error["type"] == "RuntimeError"
        assert records[1].run_result.error["message"] == "repeat-boom"


class TestJSONCompatibility:
    """Test that all records can be serialized to JSON."""

    def test_all_records_json_serializable(self, tmp_path: Path):
        """All returned records can be serialized."""
        seeds = [42, 43, 44]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        records = run_repeated(
            experiment=EXPERIMENT_B,
            task=TASK,
            seeds=seeds,
            run_factory=factory,
        )

        for record in records:
            data = record.model_dump(mode="json")
            assert isinstance(data, dict)
            assert isinstance(data["seed"], int)
            assert isinstance(data["task"], str)


class TestInputValidation:
    """Test input validation for run_repeated."""

    def test_non_callable_factory_rejected(self, tmp_path: Path):
        """Non-callable factory raises TypeError."""
        with pytest.raises(TypeError, match="run_factory must be callable"):
            run_repeated(
                experiment=EXPERIMENT_B,
                task=TASK,
                seeds=[42],
                run_factory="not_callable",  # type: ignore[arg-type]
            )

    def test_empty_task_rejected(self, tmp_path: Path):
        """Empty task raises ValueError."""

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        with pytest.raises(ValueError, match="task must be a non-empty str"):
            run_repeated(
                experiment=EXPERIMENT_B,
                task="",
                seeds=[42],
                run_factory=factory,
            )

    def test_wrong_experiment_type_rejected(self, tmp_path: Path):
        """Non-ExperimentConfig raises TypeError."""

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return _build_context(tmp_path, run_index=run_index, seed=seed), _build_agents()

        with pytest.raises(TypeError, match="experiment must be an ExperimentConfig"):
            run_repeated(
                experiment={"name": "B"},  # type: ignore[arg-type]
                task=TASK,
                seeds=[42],
                run_factory=factory,
            )

    def test_factory_wrong_return_type_rejected(self, tmp_path: Path):
        """Factory returning wrong type raises TypeError."""
        seeds = [42]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            return "wrong"  # type: ignore[return-value]

        with pytest.raises(TypeError, match="run_factory must return tuple"):
            run_repeated(
                experiment=EXPERIMENT_B,
                task=TASK,
                seeds=seeds,
                run_factory=factory,
            )

    def test_factory_wrong_tuple_length_rejected(self, tmp_path: Path):
        """Factory returning tuple of wrong length raises TypeError."""
        seeds = [42]

        def factory(run_index: int, seed: int) -> tuple[RunContext, AgentRegistry]:
            ctx = _build_context(tmp_path, run_index=run_index, seed=seed)
            return (ctx,)  # type: ignore[return-value]

        with pytest.raises(TypeError, match="tuple of length 2"):
            run_repeated(
                experiment=EXPERIMENT_B,
                task=TASK,
                seeds=seeds,
                run_factory=factory,
            )
