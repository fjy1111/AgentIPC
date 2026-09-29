from __future__ import annotations

from pathlib import Path

import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router
from agentipc.sandbox.models import SandboxResult
from agentipc.sandbox.python_runner import PythonSandbox


TASK = 'print("fresh-codeact")'
FINAL_ANSWER = "CodeAct memory reuse completed."
HISTORICAL_MEMORY_ID = "mem-historical-codeact"
EMBEDDING_DIM = 64

KNOWLEDGE = [
    {
        "document_id": "codeact-memory",
        "text": "fresh-codeact tasks can be executed by the AgentIPC sandbox.",
        "keywords": ["fresh-codeact", "sandbox", "memory"],
    }
]


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"forbidden infrastructure access: {name}")


def _agents() -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent(),
        SummarizerAgent(),
    ]:
        registry.register(agent)
    return registry


def _provider_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={"fresh-codeact": FINAL_ANSWER},
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(dim=EMBEDDING_DIM),
    )


def _service(root: Path) -> tuple[SQLiteMemoryStore, MemoryService]:
    store = SQLiteMemoryStore(root)
    provider = HashEmbeddingProvider(dim=EMBEDDING_DIM)
    return store, MemoryService(store, provider, VectorIndex(provider.dim))


def _cached_output(**updates: object) -> dict[str, object]:
    output: dict[str, object] = {
        "exit_code": 0,
        "stdout": '{"value":42}\n',
        "stderr": "",
        "timed_out": False,
        "duration_ms": 1.0,
    }
    output.update(updates)
    return output


def _seed_codeact_memory(
    service: MemoryService,
    *,
    output: dict[str, object],
) -> None:
    service.write(
        MemoryRecord(
            memory_id=HISTORICAL_MEMORY_ID,
            source_agent="summarizer",
            task_topic=TASK,
            summary=TASK,
            memory_type=MemoryType.RESULT,
            payload={
                "answer": FINAL_ANSWER,
                "execution": {
                    "operation": "codeact",
                    "output": output,
                },
            },
        )
    )


def _context(
    tmp_path: Path,
    *,
    name: str,
    service: MemoryService,
) -> tuple[MetricsCollector, RunContext]:
    metrics = MetricsCollector()
    poison = PoisonService()
    return metrics, RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(),
        registry=CapabilityRegistry(),
        state_hub=poison,  # type: ignore[arg-type]
        artifact_store=poison,  # type: ignore[arg-type]
        memory_service=service,
        metrics=metrics,
        trace_logger=TraceLogger(tmp_path / f"{name}.jsonl"),
        provider_bundle=_provider_bundle(),
        use_state=False,
        use_memory=True,
        use_sandbox=True,
    )


def test_successful_historical_codeact_reuses_output_without_sandbox(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, service = _service(tmp_path / "successful-memory")
    try:
        cached_output = _cached_output()
        _seed_codeact_memory(service, output=cached_output)

        def forbidden_run(*args, **kwargs):
            raise AssertionError("PythonSandbox.run must not be called")

        monkeypatch.setattr(PythonSandbox, "run", forbidden_run)
        metrics, ctx = _context(tmp_path, name="successful-reuse", service=service)

        final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert final.result is not None
        assert final.result["answer"] == FINAL_ANSWER

        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 1
        assert snapshot.memory_effective == 1
        assert snapshot.memory_harmful == 0
        assert snapshot.tool_call_count == 0

        historical = service.get(HISTORICAL_MEMORY_ID)
        assert historical is not None
        assert historical.reuse_count == 1
        assert historical.success_count == 1
        assert historical.failure_count == 0

        current = service.get("mem_task-successful-reuse")
        assert current is not None
        assert current.payload["execution"] == {
            "operation": "identity",
            "output": cached_output,
        }
    finally:
        store.close()


@pytest.mark.parametrize(
    "cached_output",
    [
        pytest.param(_cached_output(exit_code=1), id="nonzero-exit"),
        pytest.param(_cached_output(timed_out=True), id="timed-out"),
    ],
)
def test_failed_historical_codeact_runs_sandbox_instead(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cached_output: dict[str, object],
) -> None:
    store, service = _service(tmp_path / "failed-memory")
    try:
        _seed_codeact_memory(service, output=cached_output)
        calls: list[tuple[str, float]] = []

        def fake_run(
            self: PythonSandbox,
            code: str,
            *,
            timeout_sec: float,
        ) -> SandboxResult:
            calls.append((code, timeout_sec))
            return SandboxResult(
                exit_code=0,
                stdout="fresh-codeact\n",
                stderr="",
                timed_out=False,
                duration_ms=2.0,
            )

        monkeypatch.setattr(PythonSandbox, "run", fake_run)
        metrics, ctx = _context(tmp_path, name="failed-reuse", service=service)

        final = Orchestrator(Router(_agents())).run_task(task=TASK, ctx=ctx)

        assert final.result is not None
        assert final.result["answer"] == FINAL_ANSWER
        assert calls == [(TASK, 2.0)]

        snapshot = metrics.snapshot()
        assert snapshot.memory_retrieved == 1
        assert snapshot.memory_used == 0
        assert snapshot.memory_effective == 0
        assert snapshot.memory_harmful == 0
        assert snapshot.tool_call_count == 1

        historical = service.get(HISTORICAL_MEMORY_ID)
        assert historical is not None
        assert historical.reuse_count == 0
        assert historical.success_count == 0
        assert historical.failure_count == 0
    finally:
        store.close()
