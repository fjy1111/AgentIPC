from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.experiment import (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_D,
    ExperimentConfig,
)
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.runner import RawRunRecord, run_single
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext
from agentipc.state.hub import StateHub


_DEMO_TASK = "Summarize the AgentIPC demo evidence."
_DEMO_ANSWER = "AgentIPC demo evidence summarized deterministically."
_DEMO_KNOWLEDGE = [
    {
        "document_id": "demo-evidence",
        "text": (
            "AgentIPC demonstrates multi-agent structured communication, "
            "state exchange, artifacts, and shared memory infrastructure."
        ),
        "keywords": ["AgentIPC", "demo", "evidence"],
    }
]
_EXPERIMENTS = (EXPERIMENT_A, EXPERIMENT_B, EXPERIMENT_D)


@dataclass(frozen=True, slots=True)
class DemoReport:
    provider: str
    records: tuple[RawRunRecord, ...]


def _provider_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(default_text=_DEMO_ANSWER),
        embedding=HashEmbeddingProvider(dim=64),
    )


def _agent_registry() -> AgentRegistry:
    agents = AgentRegistry()
    agents.register(PlannerAgent())
    agents.register(RetrieverAgent(knowledge=_DEMO_KNOWLEDGE))
    agents.register(ExecutorAgent())
    agents.register(SummarizerAgent())
    return agents


def _run_experiment(
    *,
    experiment: ExperimentConfig,
    root: Path,
    config: AgentIPCConfig,
) -> RawRunRecord:
    provider_bundle = _provider_bundle()
    memory_store = SQLiteMemoryStore(root / "memory")
    state_hub = StateHub(transport="inproc")
    try:
        memory_service = MemoryService(
            memory_store,
            provider_bundle.embedding,
            VectorIndex(provider_bundle.embedding.dim),
        )
        ctx = RunContext(
            trace_id=f"demo-{experiment.name.value}-trace",
            task_id=f"demo-{experiment.name.value}-task",
            mode=experiment.mode,
            config=config,
            registry=CapabilityRegistry(),
            state_hub=state_hub,
            artifact_store=ArtifactStore(root / "artifacts"),
            memory_service=memory_service,
            metrics=MetricsCollector(),
            trace_logger=TraceLogger(root / "trace.jsonl"),
            provider_bundle=provider_bundle,
            use_state=experiment.use_state,
            use_memory=experiment.use_memory,
            use_sandbox=False,
        )
        return run_single(
            experiment=experiment,
            task=_DEMO_TASK,
            ctx=ctx,
            agent_registry=_agent_registry(),
        )
    finally:
        state_hub.close()
        memory_store.close()


def run_demo(*, provider: str = "mock") -> DemoReport:
    if type(provider) is not str:
        raise TypeError("provider must be a str")
    if provider != "mock":
        raise ValueError("demo currently supports only provider='mock'")

    config = AgentIPCConfig(
        llm_provider="mock",
        embedding_provider="hash",
        random_seed=42,
    )
    records: list[RawRunRecord] = []
    with tempfile.TemporaryDirectory(prefix="agentipc-demo-") as temp_dir:
        root = Path(temp_dir)
        for experiment in _EXPERIMENTS:
            records.append(
                _run_experiment(
                    experiment=experiment,
                    root=root / experiment.name.value,
                    config=config,
                )
            )

    if not all(record.run_result.success for record in records):
        raise RuntimeError("AgentIPC demo runtime did not complete successfully")
    answers = {record.run_result.answer for record in records}
    if len(answers) != 1:
        raise RuntimeError("AgentIPC demo A/B/D final answers differ")

    return DemoReport(provider=provider, records=tuple(records))


def _mode_label(value: str) -> str:
    if value == "text":
        return "Text"
    if value == "structured":
        return "Structured"
    return value


def render_demo(report: DemoReport) -> str:
    if not isinstance(report, DemoReport):
        raise TypeError("report must be a DemoReport")

    lines = [
        f"AgentIPC offline demo (provider={report.provider})",
        (
            "A/B use the same task/provider/agents/seed; "
            "the communication representation differs."
        ),
        "This single demo is descriptive and is not a benchmark conclusion.",
    ]

    metric_names = (
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
        "latency_ms",
    )

    for record in report.records:
        result = record.run_result
        metrics = result.metrics
        experiment = record.experiment
        lines.append("")
        lines.append(
            f"Experiment {experiment.name.value} - "
            f"{_mode_label(experiment.mode.value)}"
        )
        lines.append(
            "  "
            f"success={result.success} "
            f"use_state={experiment.use_state} "
            f"use_memory={experiment.use_memory} "
            f"use_sandbox={record.use_sandbox}"
        )
        lines.append(f"  final_answer={result.answer}")
        lines.append(
            "  "
            + " ".join(
                f"{name}={metrics[name]}"
                for name in metric_names
            )
        )

    lines.extend(
        [
            "",
            (
                "D demonstrates the structured full-system path with State and "
                "Memory enabled. A fresh one-shot D run may have memory_used=0."
            ),
        ]
    )
    return "\n".join(lines)
