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
from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.memory.sqlite_store import SQLiteMemoryStore
from agentipc.memory.vector_index import VectorIndex
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext
from agentipc.scenarios.codeact_eval import (
    CodeActEvaluation,
    evaluate_codeact_execution,
)
from agentipc.scenarios.models import CodeActTask
from agentipc.state.hub import StateHub


class CodeActRoundResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: CodeActTask
    run_record: RawRunRecord
    execution: dict[str, object]
    evaluation: CodeActEvaluation


def run_codeact_chain(
    *,
    tasks: list[CodeActTask],
    fixture_root: str | Path,
    work_root: str | Path,
    config: AgentIPCConfig,
    provider_bundle: ProviderBundle,
) -> list[CodeActRoundResult]:
    _validate_inputs(
        tasks=tasks,
        fixture_root=fixture_root,
        work_root=work_root,
        config=config,
        provider_bundle=provider_bundle,
    )
    _validate_task_sequence(tasks)

    resolved_fixture_root = _validate_fixture_root(fixture_root)
    runtime_tasks = [
        _build_runtime_task(task, resolved_fixture_root)
        for task in tasks
    ]

    root = Path(work_root)
    if root.exists() and not root.is_dir():
        raise ValueError("work_root must be a directory path")
    root.mkdir(parents=True, exist_ok=True)

    memory_store = SQLiteMemoryStore(root / "memory")
    try:
        memory_service = MemoryService(
            memory_store,
            provider_bundle.embedding,
            VectorIndex(provider_bundle.embedding.dim),
        )
        results: list[CodeActRoundResult] = []

        for task, runtime_task in zip(tasks, runtime_tasks, strict=True):
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
                agents.register(RetrieverAgent(knowledge=[]))
                agents.register(ExecutorAgent())
                agents.register(SummarizerAgent())

                ctx = RunContext(
                    trace_id=f"codeact-{task.group_id}-round-{task.round}-trace",
                    task_id=f"codeact-{task.group_id}-round-{task.round}",
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
                    use_sandbox=True,
                )

                record = run_single(
                    experiment=EXPERIMENT_D,
                    task=runtime_task,
                    ctx=ctx,
                    agent_registry=agents,
                )
                execution = _load_round_execution(
                    memory_service=memory_service,
                    memory_id=f"mem_{ctx.task_id}",
                    runtime_task=runtime_task,
                    run_record=record,
                )
                evaluation = evaluate_codeact_execution(task, execution)
                if evaluation.error == "invalid_execution":
                    raise RuntimeError(
                        "CodeAct RESULT memory contains invalid execution"
                    )
                results.append(
                    CodeActRoundResult(
                        task=task,
                        run_record=record,
                        execution=execution,
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
    fixture_root: object,
    work_root: object,
    config: object,
    provider_bundle: object,
) -> None:
    if type(tasks) is not list:
        raise TypeError("tasks must be a list[CodeActTask]")
    if not tasks:
        raise ValueError("tasks must not be empty")
    if not all(isinstance(task, CodeActTask) for task in tasks):
        raise TypeError("tasks must contain only CodeActTask values")

    if not isinstance(fixture_root, (str, Path)):
        raise TypeError("fixture_root must be str or pathlib.Path")
    if not isinstance(work_root, (str, Path)):
        raise TypeError("work_root must be str or pathlib.Path")
    if not isinstance(config, AgentIPCConfig):
        raise TypeError("config must be an AgentIPCConfig")
    if not isinstance(provider_bundle, ProviderBundle):
        raise TypeError("provider_bundle must be a ProviderBundle")


def _validate_task_sequence(tasks: list[CodeActTask]) -> None:
    group_id = tasks[0].group_id
    previous_round = tasks[0].round

    for task in tasks[1:]:
        if task.group_id != group_id:
            raise ValueError("all tasks must have the same group_id")
        if task.round <= previous_round:
            raise ValueError("task rounds must be strictly increasing")
        previous_round = task.round


def _validate_fixture_root(fixture_root: str | Path) -> Path:
    root = Path(fixture_root)
    if not root.exists() or not root.is_dir():
        raise ValueError("fixture_root must be an existing directory")
    return root.resolve()


def _build_runtime_task(task: CodeActTask, fixture_root: Path) -> str:
    contents: dict[str, str] = {}
    for artifact in sorted(task.input_artifacts):
        artifact_path = _resolve_input_artifact(fixture_root, artifact)
        contents[artifact] = artifact_path.read_text(encoding="utf-8")

    prelude = f"INPUT_ARTIFACTS = {contents!r}\n\n"
    return prelude + task.code


def _resolve_input_artifact(fixture_root: Path, artifact: str) -> Path:
    if type(artifact) is not str:
        raise TypeError("input_artifacts items must be str")
    if artifact == "" or artifact.startswith("/") or "\\" in artifact:
        raise ValueError("input_artifact must be a portable relative path")

    parts = artifact.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("input_artifact must be a portable relative path")
    if len(parts[0]) >= 2 and parts[0][1] == ":":
        raise ValueError("input_artifact must be a portable relative path")

    candidate = fixture_root.joinpath(*parts)
    if not candidate.exists():
        raise ValueError(f"input artifact does not exist: {artifact}")
    if candidate.is_symlink():
        raise ValueError(f"input artifact must be a regular file: {artifact}")

    resolved = candidate.resolve()
    try:
        resolved.relative_to(fixture_root)
    except ValueError as exc:
        raise ValueError(f"input artifact escapes fixture_root: {artifact}") from exc

    if not resolved.is_file():
        raise ValueError(f"input artifact must be a regular file: {artifact}")
    return resolved


def _load_round_execution(
    *,
    memory_service: MemoryService,
    memory_id: str,
    runtime_task: str,
    run_record: RawRunRecord,
) -> dict[str, object]:
    if not run_record.run_result.success:
        raise RuntimeError("CodeAct runtime round did not complete successfully")

    record = memory_service.get(memory_id)
    if record is None:
        raise RuntimeError(
            "successful CodeAct round did not write its RESULT memory"
        )

    _validate_round_memory(record, runtime_task=runtime_task)
    execution = record.payload.get("execution")
    if type(execution) is not dict:
        raise RuntimeError("CodeAct RESULT memory contains invalid execution")
    return execution


def _validate_round_memory(record: MemoryRecord, *, runtime_task: str) -> None:
    if record.memory_type is not MemoryType.RESULT:
        raise RuntimeError("CodeAct round memory must have type RESULT")
    if record.task_topic != runtime_task:
        raise RuntimeError(
            "CodeAct round memory task_topic does not match runtime task"
        )
