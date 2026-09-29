"""Tests for single-run benchmark adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

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
from agentipc.evaluation.runner import RawRunRecord, run_single
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import MessageStatus
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
    {
        "document_id": "system-info",
        "text": "Use systemctl for service management.",
        "keywords": ["systemctl", "service"],
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


def _build_memory_service(tmp_path: Path) -> MemoryService:
    """Build memory service with fresh SQLite store."""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    embedding = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    return MemoryService(store, embedding, VectorIndex(embedding.dim))


def _build_context(
    tmp_path: Path,
    *,
    mode: RunMode,
    use_state: bool,
    use_memory: bool,
    use_sandbox: bool,
) -> RunContext:
    """Build a fresh RunContext for testing."""
    trace_path = tmp_path / "trace.jsonl"
    return RunContext(
        trace_id="trace-test",
        task_id="task-test",
        mode=mode,
        config=AgentIPCConfig(random_seed=42),
        registry=CapabilityRegistry(),
        state_hub=StateHub(transport="inproc"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        memory_service=_build_memory_service(tmp_path),
        metrics=MetricsCollector(),
        trace_logger=TraceLogger(trace_path),
        provider_bundle=_build_provider_bundle(),
        use_state=use_state,
        use_memory=use_memory,
        use_sandbox=use_sandbox,
    )


class TestRunSingleBasic:
    """Test basic run_single execution with all four experiments."""

    @pytest.mark.parametrize(
        "experiment",
        [EXPERIMENT_A, EXPERIMENT_B, EXPERIMENT_C, EXPERIMENT_D],
        ids=["A", "B", "C", "D"],
    )
    def test_experiment_runs_successfully(
        self,
        tmp_path: Path,
        experiment: ExperimentConfig,
    ):
        """Each experiment configuration produces a successful raw record."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        record = run_single(
            experiment=experiment,
            task=TASK,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        assert isinstance(record, RawRunRecord)
        assert record.experiment == experiment
        assert record.task == TASK
        assert record.seed == 42
        assert record.use_sandbox is False
        assert record.run_result.success is True
        assert record.run_result.answer == EXPECTED_ANSWER
        assert record.run_result.error is None
        assert Path(record.run_result.trace_path).exists()


class TestTaskHash:
    """Test that task_hash is computed correctly."""

    def test_task_hash_is_sha256_of_utf8_bytes(self, tmp_path: Path):
        """task_hash must be SHA-256(task.encode('utf-8'))."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        record = run_single(
            experiment=EXPERIMENT_B,
            task=TASK,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        expected_hash = hashlib.sha256(TASK.encode("utf-8")).hexdigest()
        assert record.task_hash == expected_hash

    def test_unicode_task_hash_computed_correctly(self, tmp_path: Path):
        """Unicode tasks produce correct UTF-8 based hash."""
        unicode_task = "诊断 openEuler 网络连接"
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        record = run_single(
            experiment=EXPERIMENT_B,
            task=unicode_task,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        expected_hash = hashlib.sha256(unicode_task.encode("utf-8")).hexdigest()
        assert record.task_hash == expected_hash


class TestExperimentMetrics:
    """Test that A/B/C/D produce expected metric patterns."""

    def test_experiment_a_text_metrics(self, tmp_path: Path):
        """A: text mode with text_chars > 0, no protocol_bytes/state/memory."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        record = run_single(
            experiment=EXPERIMENT_A,
            task=TASK,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        metrics = record.run_result.metrics
        assert metrics["text_chars"] > 0
        assert metrics["protocol_bytes"] == 0
        assert metrics["state_transfer_count"] == 0
        assert metrics["memory_retrieved"] == 0
        assert metrics["message_count"] == 8
        assert metrics["success"] is True

    def test_experiment_b_structured_metrics(self, tmp_path: Path):
        """B: structured mode with protocol_bytes > 0, no state/memory."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        record = run_single(
            experiment=EXPERIMENT_B,
            task=TASK,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        metrics = record.run_result.metrics
        assert metrics["protocol_bytes"] > 0
        assert metrics["state_transfer_count"] == 0
        assert metrics["memory_retrieved"] == 0
        assert metrics["message_count"] == 8
        assert metrics["success"] is True

    def test_experiment_c_state_metrics(self, tmp_path: Path):
        """C: structured + state with state_transfer_count > 0, no memory."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        record = run_single(
            experiment=EXPERIMENT_C,
            task=TASK,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        metrics = record.run_result.metrics
        assert metrics["protocol_bytes"] > 0
        assert metrics["state_transfer_count"] > 0
        assert metrics["state_bytes"] > 0
        assert metrics["memory_retrieved"] == 0
        assert metrics["message_count"] == 8
        assert metrics["success"] is True

    def test_experiment_d_full_system_metrics(self, tmp_path: Path):
        """D: full system with state + memory write (first run has no hits)."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        record = run_single(
            experiment=EXPERIMENT_D,
            task=TASK,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        metrics = record.run_result.metrics
        assert metrics["protocol_bytes"] > 0
        assert metrics["state_transfer_count"] > 0
        assert metrics["state_bytes"] > 0
        # First run: no historical memory
        assert metrics["memory_retrieved"] == 0
        assert metrics["message_count"] == 8
        assert metrics["success"] is True

        # Verify memory was written
        memory_service = ctx.memory_service
        memory_id = f"mem_{ctx.task_id}"
        stored = memory_service.get(memory_id)
        assert stored is not None
        assert stored.task_topic == TASK


class TestOriginalContextImmutable:
    """Test that run_single does not modify the original context."""

    def test_original_context_flags_unchanged(self, tmp_path: Path):
        """Original ctx mode/use_state/use_memory/use_sandbox must not change."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        # Record original values
        original_mode = ctx.mode
        original_use_state = ctx.use_state
        original_use_memory = ctx.use_memory
        original_use_sandbox = ctx.use_sandbox

        # Run with different experiment
        run_single(
            experiment=EXPERIMENT_C,
            task=TASK,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        # Verify original context unchanged
        assert ctx.mode is original_mode
        assert ctx.use_state is original_use_state
        assert ctx.use_memory is original_use_memory
        assert ctx.use_sandbox is original_use_sandbox


class TestSandboxPreserved:
    """Test that use_sandbox is preserved from ctx, not from experiment."""

    def test_sandbox_enabled_preserved(self, tmp_path: Path):
        """When ctx.use_sandbox=True, record.use_sandbox must be True."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=True,
        )
        agent_registry = _build_agents()

        # Simple print task for sandbox execution
        code_task = 'print("agentipc-benchmark-codeact")'

        record = run_single(
            experiment=EXPERIMENT_B,
            task=code_task,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        assert record.use_sandbox is True
        assert record.run_result.success is True
        # Verify sandbox was actually used
        metrics = record.run_result.metrics
        assert metrics["tool_call_count"] == 1


class TestRuntimeFailureCapture:
    """Test that runtime exceptions are captured as failed records."""

    def test_runtime_error_produces_failed_record(self, tmp_path: Path):
        """Runtime errors during execution produce success=False record."""

        class FailingPlannerAgent(PlannerAgent):
            """Planner that always fails during handle."""

            def handle(
                self,
                envelope: AgentEnvelope,
                ctx: RunContext,
            ) -> AgentEnvelope:
                raise RuntimeError("single-run-boom")

        # Build registry with failing planner
        registry = AgentRegistry()
        registry.register(FailingPlannerAgent())
        registry.register(RetrieverAgent(KNOWLEDGE))
        registry.register(ExecutorAgent())
        registry.register(SummarizerAgent())

        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )

        # Should not raise, should return failed record
        record = run_single(
            experiment=EXPERIMENT_B,
            task=TASK,
            ctx=ctx,
            agent_registry=registry,
        )

        assert record.run_result.success is False
        assert record.run_result.answer == ""
        assert record.run_result.error is not None
        assert record.run_result.error["type"] == "RuntimeError"
        assert record.run_result.error["message"] == "single-run-boom"
        assert record.run_result.metrics["success"] is False
        assert record.run_result.metrics["latency_ms"] >= 0
        # Trace still captured
        assert Path(record.run_result.trace_path).exists()


class TestJSONSerialization:
    """Test that RawRunRecord can be serialized to JSON."""

    def test_raw_record_json_compatible(self, tmp_path: Path):
        """RawRunRecord.model_dump(mode='json') produces JSON-compatible dict."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        record = run_single(
            experiment=EXPERIMENT_B,
            task=TASK,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        data = record.model_dump(mode="json")

        # Verify structure
        assert isinstance(data, dict)
        assert "experiment" in data
        assert "task" in data
        assert "task_hash" in data
        assert "seed" in data
        assert "use_sandbox" in data
        assert "run_result" in data

        # Verify types are JSON-compatible
        assert isinstance(data["experiment"], dict)
        assert isinstance(data["task"], str)
        assert isinstance(data["task_hash"], str)
        assert isinstance(data["seed"], int)
        assert isinstance(data["use_sandbox"], bool)
        assert isinstance(data["run_result"], dict)


class TestInputValidation:
    """Test input validation for run_single."""

    def test_empty_task_rejected(self, tmp_path: Path):
        """Empty task string must be rejected."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        with pytest.raises(ValueError, match="task must be a non-empty str"):
            run_single(
                experiment=EXPERIMENT_B,
                task="",
                ctx=ctx,
                agent_registry=agent_registry,
            )

    def test_wrong_experiment_type_rejected(self, tmp_path: Path):
        """Non-ExperimentConfig experiment must be rejected."""
        ctx = _build_context(
            tmp_path,
            mode=RunMode.STRUCTURED,
            use_state=False,
            use_memory=False,
            use_sandbox=False,
        )
        agent_registry = _build_agents()

        with pytest.raises(TypeError, match="experiment must be an ExperimentConfig"):
            run_single(
                experiment={"name": "B"},  # type: ignore[arg-type]
                task=TASK,
                ctx=ctx,
                agent_registry=agent_registry,
            )

    def test_wrong_ctx_type_rejected(self, tmp_path: Path):
        """Non-RunContext ctx must be rejected."""
        agent_registry = _build_agents()

        with pytest.raises(TypeError, match="ctx must be a RunContext"):
            run_single(
                experiment=EXPERIMENT_B,
                task=TASK,
                ctx={},  # type: ignore[arg-type]
                agent_registry=agent_registry,
            )
