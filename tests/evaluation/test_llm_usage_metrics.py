"""Integration tests for LLM usage metrics (B066-SA1).

These tests verify that real LLM API usage (prompt_tokens, completion_tokens,
latency_ms) flows correctly from LLMResponse through MetricsCollector into
the final run_result.metrics, independent of communication metrics.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.experiment import EXPERIMENT_B
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.runner import run_single
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.base import LLMResponse
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.state.hub import StateHub


TASK = "test task for llm usage metrics"
EMBEDDING_DIM = 128

KNOWLEDGE = [
    {
        "document_id": "doc-1",
        "text": "test document for retrieval",
        "keywords": ["test"],
    }
]


class DeterministicTestLLMProvider:
    """
    Local deterministic fake LLM provider for testing real-like usage metrics.
    Does not call any network. Returns fixed responses with realistic token counts.
    """

    def __init__(self) -> None:
        self.call_count = 0
        self.responses = [
            LLMResponse(
                text="planner response",
                prompt_tokens=11,
                completion_tokens=3,
                latency_ms=1.25,
                raw={"provider": "test"},
            ),
            LLMResponse(
                text="final response",
                prompt_tokens=20,
                completion_tokens=4,
                latency_ms=2.5,
                raw={"provider": "test"},
            ),
        ]

    def complete(self, messages, *, temperature=0.0):
        if self.call_count >= len(self.responses):
            raise RuntimeError("test provider exhausted")
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


def _build_agents() -> AgentRegistry:
    """Build agent registry with all four required agents."""
    registry = AgentRegistry()
    registry.register(PlannerAgent())
    registry.register(RetrieverAgent(KNOWLEDGE))
    registry.register(ExecutorAgent())
    registry.register(SummarizerAgent())
    return registry


def _build_memory_service(tmp_path: Path) -> MemoryService:
    """Build memory service with fresh SQLite store."""
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    embedding = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    return MemoryService(store, embedding, VectorIndex(embedding.dim))


def _build_context(
    tmp_path: Path,
    provider_bundle: ProviderBundle,
) -> RunContext:
    """Build a fresh RunContext for testing."""
    trace_path = tmp_path / "trace.jsonl"
    return RunContext(
        trace_id="trace-llm-metrics",
        task_id="task-llm-metrics",
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(random_seed=42),
        registry=CapabilityRegistry(),
        state_hub=StateHub(transport="inproc"),
        artifact_store=ArtifactStore(tmp_path / "artifacts"),
        memory_service=_build_memory_service(tmp_path),
        metrics=MetricsCollector(),
        trace_logger=TraceLogger(trace_path),
        provider_bundle=provider_bundle,
        use_state=False,
        use_memory=False,
        use_sandbox=False,
    )


def test_llm_usage_metrics_integration_with_real_like_provider(tmp_path: Path):
    """
    Full integration test with deterministic provider that reports real-like usage.
    Verifies that LLM metrics flow through the complete pipeline.
    """
    llm = DeterministicTestLLMProvider()
    embedding = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    provider_bundle = ProviderBundle(llm=llm, embedding=embedding)

    ctx = _build_context(tmp_path, provider_bundle)
    agent_registry = _build_agents()

    record = run_single(
        experiment=EXPERIMENT_B,
        task=TASK,
        ctx=ctx,
        agent_registry=agent_registry,
    )

    assert record.run_result.success is True
    assert record.run_result.answer == "final response"

    metrics = record.run_result.metrics
    assert metrics["llm_call_count"] == 2
    assert metrics["llm_prompt_tokens"] == 31
    assert metrics["llm_completion_tokens"] == 7
    assert metrics["llm_total_tokens"] == 38
    assert metrics["llm_usage_missing_count"] == 0
    assert metrics["llm_latency_ms"] == 3.75

    # Communication metrics still exist
    assert metrics["message_count"] == 8
    assert metrics["protocol_bytes"] > 0


def test_llm_usage_metrics_with_mock_provider_reports_missing(tmp_path: Path):
    """
    Verify that Mock provider (no real usage) correctly reports missing metrics.
    """
    llm = MockLLMProvider(default_text="mock answer")
    embedding = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    provider_bundle = ProviderBundle(llm=llm, embedding=embedding)

    ctx = _build_context(tmp_path, provider_bundle)
    agent_registry = _build_agents()

    record = run_single(
        experiment=EXPERIMENT_B,
        task=TASK,
        ctx=ctx,
        agent_registry=agent_registry,
    )

    assert record.run_result.success is True

    metrics = record.run_result.metrics
    assert metrics["llm_call_count"] == 2
    assert metrics["llm_prompt_tokens"] == 0
    assert metrics["llm_completion_tokens"] == 0
    assert metrics["llm_total_tokens"] == 0
    assert metrics["llm_usage_missing_count"] == 2
    assert metrics["llm_latency_ms"] == 0.0


def test_communication_metrics_remain_independent_of_llm_metrics(tmp_path: Path):
    """
    Verify that adding LLM usage metrics does not interfere with
    existing communication metrics like message_count and protocol_bytes.
    """
    llm = DeterministicTestLLMProvider()
    embedding = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    provider_bundle = ProviderBundle(llm=llm, embedding=embedding)

    ctx = _build_context(tmp_path, provider_bundle)
    agent_registry = _build_agents()

    record = run_single(
        experiment=EXPERIMENT_B,
        task=TASK,
        ctx=ctx,
        agent_registry=agent_registry,
    )

    metrics = record.run_result.metrics

    # LLM metrics correctly recorded
    assert metrics["llm_call_count"] == 2
    assert metrics["llm_prompt_tokens"] == 31
    assert metrics["llm_total_tokens"] == 38

    # Communication metrics still present and correct
    assert metrics["message_count"] == 8
    assert metrics["text_chars"] == 0
    assert metrics["protocol_bytes"] > 0
    assert "text_tokens" in metrics
