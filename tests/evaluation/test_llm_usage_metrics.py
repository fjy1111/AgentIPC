from types import SimpleNamespace

import pytest

from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.runner import run_single
from agentipc.providers.base import LLMResponse
from agentipc.providers.hash_embedding import HashEmbeddingProvider


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


def test_llm_usage_metrics_integration_with_real_like_provider() -> None:
    """
    Full integration test with deterministic provider that reports real-like usage.
    Verifies that LLM metrics flow through the complete pipeline.
    """
    llm = DeterministicTestLLMProvider()
    embedding = HashEmbeddingProvider(dim=128)
    provider_bundle = SimpleNamespace(llm=llm, embedding=embedding)

    config = {
        "mode": "text",
        "use_state": False,
        "use_memory": False,
        "use_sandbox": False,
    }
    knowledge_base = [
        {
            "document_id": "doc-1",
            "text": "test document",
        }
    ]

    run_result = run_single(
        task="test task",
        provider_bundle=provider_bundle,
        config=config,
        knowledge_base=knowledge_base,
    )

    assert run_result.success is True
    assert run_result.answer == "final response"

    metrics = run_result.metrics
    assert metrics["llm_call_count"] == 2
    assert metrics["llm_prompt_tokens"] == 31
    assert metrics["llm_completion_tokens"] == 7
    assert metrics["llm_total_tokens"] == 38
    assert metrics["llm_usage_missing_count"] == 0
    assert metrics["llm_latency_ms"] == 3.75

    assert metrics["message_count"] > 0
    assert "text_chars" in metrics


def test_llm_usage_metrics_with_mock_provider_reports_missing() -> None:
    """
    Verify that Mock provider (no real usage) correctly reports missing metrics.
    """
    from agentipc.providers.mock_llm import MockLLMProvider

    llm = MockLLMProvider(default_text="mock answer")
    embedding = HashEmbeddingProvider(dim=128)
    provider_bundle = SimpleNamespace(llm=llm, embedding=embedding)

    config = {
        "mode": "text",
        "use_state": False,
        "use_memory": False,
        "use_sandbox": False,
    }
    knowledge_base = [
        {
            "document_id": "doc-1",
            "text": "test document",
        }
    ]

    run_result = run_single(
        task="test task",
        provider_bundle=provider_bundle,
        config=config,
        knowledge_base=knowledge_base,
    )

    assert run_result.success is True

    metrics = run_result.metrics
    assert metrics["llm_call_count"] == 2
    assert metrics["llm_prompt_tokens"] == 0
    assert metrics["llm_completion_tokens"] == 0
    assert metrics["llm_total_tokens"] == 0
    assert metrics["llm_usage_missing_count"] == 2
    assert metrics["llm_latency_ms"] == 0.0


def test_communication_metrics_remain_independent_of_llm_metrics() -> None:
    """
    Verify that adding LLM usage metrics does not interfere with
    existing communication metrics like message_count and text_chars.
    """
    llm = DeterministicTestLLMProvider()
    embedding = HashEmbeddingProvider(dim=128)
    provider_bundle = SimpleNamespace(llm=llm, embedding=embedding)

    config = {
        "mode": "text",
        "use_state": False,
        "use_memory": False,
        "use_sandbox": False,
    }
    knowledge_base = [
        {
            "document_id": "doc-1",
            "text": "test document",
        }
    ]

    run_result = run_single(
        task="test task",
        provider_bundle=provider_bundle,
        config=config,
        knowledge_base=knowledge_base,
    )

    metrics = run_result.metrics

    assert metrics["llm_call_count"] == 2
    assert metrics["llm_prompt_tokens"] == 31
    assert metrics["llm_total_tokens"] == 38

    assert metrics["message_count"] > 0
    assert metrics["text_chars"] > 0
    assert "text_tokens" in metrics
    assert "protocol_bytes" in metrics
