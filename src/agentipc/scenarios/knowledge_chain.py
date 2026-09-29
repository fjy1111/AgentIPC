from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.artifacts.store import ArtifactStore
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.experiment import EXPERIMENT_D
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.runner import RawRunRecord, run_single
from agentipc.evaluation.trace import TraceLogger
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext
from agentipc.scenarios.knowledge_eval import (
    KnowledgeEvaluation,
    evaluate_knowledge_answer,
)
from agentipc.scenarios.knowledge_loader import KnowledgeDocument
from agentipc.scenarios.models import KnowledgeTask
from agentipc.state.hub import StateHub


class KnowledgeRoundResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: KnowledgeTask
    run_record: RawRunRecord
    evaluation: KnowledgeEvaluation


def run_knowledge_chain(
    *,
    tasks: list[KnowledgeTask],
    documents: list[KnowledgeDocument],
    work_root: str | Path,
    config: AgentIPCConfig,
    provider_bundle: ProviderBundle,
) -> list[KnowledgeRoundResult]:
    _validate_inputs(
        tasks=tasks,
        documents=documents,
        work_root=work_root,
        config=config,
        provider_bundle=provider_bundle,
    )
    _validate_task_sequence(tasks)
    _validate_documents(tasks=tasks, documents=documents)

    root = Path(work_root)
    if root.exists() and not root.is_dir():
        raise ValueError("work_root must be a directory path")
    root.mkdir(parents=True, exist_ok=True)

    knowledge = [
        {
            "document_id": document.document_id,
            "text": document.body,
            "keywords": list(document.tags),
        }
        for document in documents
    ]

    memory_store = SQLiteMemoryStore(root / "memory")
    try:
        memory_service = MemoryService(
            memory_store,
            provider_bundle.embedding,
            VectorIndex(provider_bundle.embedding.dim),
        )
        results: list[KnowledgeRoundResult] = []

        for task in tasks:
            state_hub = StateHub(transport="inproc")
            try:
                capability_registry = CapabilityRegistry()
                artifact_store = ArtifactStore(
                    root / "artifacts" / f"round-{task.round:02d}"
                )
                metrics = MetricsCollector()
                trace_logger = TraceLogger(
                    root / "traces" / f"round-{task.round:02d}.jsonl"
                )

                agents = AgentRegistry()
                agents.register(PlannerAgent())
                agents.register(RetrieverAgent(knowledge=knowledge))
                agents.register(ExecutorAgent())
                agents.register(SummarizerAgent())

                ctx = RunContext(
                    trace_id=(
                        f"knowledge-{task.group_id}-round-{task.round}-trace"
                    ),
                    task_id=f"knowledge-{task.group_id}-round-{task.round}",
                    mode=EXPERIMENT_D.mode,
                    config=config,
                    registry=capability_registry,
                    state_hub=state_hub,
                    artifact_store=artifact_store,
                    memory_service=memory_service,
                    metrics=metrics,
                    trace_logger=trace_logger,
                    provider_bundle=provider_bundle,
                    use_state=EXPERIMENT_D.use_state,
                    use_memory=EXPERIMENT_D.use_memory,
                    use_sandbox=False,
                )

                record = run_single(
                    experiment=EXPERIMENT_D,
                    task=task.query,
                    ctx=ctx,
                    agent_registry=agents,
                )
                evaluation = evaluate_knowledge_answer(
                    task,
                    record.run_result.answer,
                )
                results.append(
                    KnowledgeRoundResult(
                        task=task,
                        run_record=record,
                        evaluation=evaluation,
                    )
                )
            finally:
                state_hub.close()

        return results
    finally:
        memory_store.close()


def _validate_inputs(
    *,
    tasks: object,
    documents: object,
    work_root: object,
    config: object,
    provider_bundle: object,
) -> None:
    if type(tasks) is not list:
        raise TypeError("tasks must be a list[KnowledgeTask]")
    if not tasks:
        raise ValueError("tasks must not be empty")
    if not all(isinstance(task, KnowledgeTask) for task in tasks):
        raise TypeError("tasks must contain only KnowledgeTask values")

    if type(documents) is not list:
        raise TypeError("documents must be a list[KnowledgeDocument]")
    if not documents:
        raise ValueError("documents must not be empty")
    if not all(isinstance(document, KnowledgeDocument) for document in documents):
        raise TypeError("documents must contain only KnowledgeDocument values")

    if not isinstance(work_root, (str, Path)):
        raise TypeError("work_root must be str or pathlib.Path")
    if not isinstance(config, AgentIPCConfig):
        raise TypeError("config must be an AgentIPCConfig")
    if not isinstance(provider_bundle, ProviderBundle):
        raise TypeError("provider_bundle must be a ProviderBundle")


def _validate_task_sequence(tasks: list[KnowledgeTask]) -> None:
    group_id = tasks[0].group_id
    previous_round = tasks[0].round

    for task in tasks[1:]:
        if task.group_id != group_id:
            raise ValueError("all tasks must have the same group_id")
        if task.round <= previous_round:
            raise ValueError("task rounds must be strictly increasing")
        previous_round = task.round


def _validate_documents(
    *,
    tasks: list[KnowledgeTask],
    documents: list[KnowledgeDocument],
) -> None:
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("document_id values must be unique")

    available = set(document_ids)
    for task in tasks:
        missing = set(task.expected.evidence_ids) - available
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(
                f"task round {task.round} references missing documents: "
                f"{missing_text}"
            )
