"""Single-run benchmark adapter for A/B/C/D experiments.

This module provides the adapter that executes one ExperimentConfig against
one task using the existing Runtime infrastructure, producing a standardized
raw run record for later aggregation and analysis.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from agentipc.evaluation.experiment import ExperimentConfig
from agentipc.evaluation.metrics import TaskTimer
from agentipc.evaluation.text_counter import TextCounter
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.bootstrap import run_handshake
from agentipc.runtime.context import RunContext
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.reference_resolver import ReferenceResolver
from agentipc.runtime.result import RunResult
from agentipc.runtime.router import Router
from agentipc.runtime.text_transport import TextTransport


class RawRunRecord(BaseModel):
    """Raw benchmark record for a single experiment run.

    Captures the experiment configuration, task, execution result, and
    infrastructure metadata for one complete Runtime execution.
    """

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig
    task: str = Field(min_length=1, strict=True)
    task_hash: str = Field(min_length=64, max_length=64, strict=True)
    seed: int = Field(ge=0, strict=True)
    use_sandbox: StrictBool
    run_result: RunResult

    @field_validator("task_hash")
    @classmethod
    def _validate_task_hash(cls, value: str) -> str:
        """Validate task_hash is a valid SHA-256 hex digest."""
        if not all(c in "0123456789abcdef" for c in value):
            raise ValueError("task_hash must be a lowercase hex string")
        return value


def run_single(
    *,
    experiment: ExperimentConfig,
    task: str,
    ctx: RunContext,
    agent_registry: AgentRegistry,
) -> RawRunRecord:
    """Execute one experiment configuration against one task.

    This function:
    1. Validates inputs
    2. Creates a fresh RunContext with experiment flags applied
    3. Runs handshake protocol
    4. Executes the full Runtime task chain
    5. Captures success or failure as a structured record

    The original ctx is never modified. Experiment flags (mode, use_state,
    use_memory) are applied to a new context. The use_sandbox flag is
    preserved from the original ctx.

    Args:
        experiment: The A/B/C/D experiment configuration to apply
        task: The task description (non-empty string)
        ctx: The base runtime context (will not be modified)
        agent_registry: Registry containing the four required agents

    Returns:
        RawRunRecord containing experiment config, task, and execution result

    Raises:
        TypeError: If arguments have wrong types
        ValueError: If task is empty or infrastructure is misconfigured
        RuntimeError: If handshake fails (indicates setup error, not task failure)
    """
    # Validate inputs
    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig")
    if type(task) is not str:
        raise TypeError("task must be a str")
    if task == "":
        raise ValueError("task must be a non-empty str")
    if not isinstance(ctx, RunContext):
        raise TypeError("ctx must be a RunContext")
    if not isinstance(agent_registry, AgentRegistry):
        raise TypeError("agent_registry must be an AgentRegistry")

    # Compute task hash (exact UTF-8 bytes, no normalization)
    task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()

    # Apply experiment configuration to a new context
    # IMPORTANT: preserve use_sandbox from original ctx
    run_ctx = replace(
        ctx,
        mode=experiment.mode,
        use_state=experiment.use_state,
        use_memory=experiment.use_memory,
    )

    # Run handshake protocol
    run_handshake(
        agent_registry=agent_registry,
        capability_registry=run_ctx.registry,
        trace_logger=run_ctx.trace_logger,
        trace_id=run_ctx.trace_id,
        task_id=run_ctx.task_id,
    )

    # Build orchestrator based on mode
    router = Router(agent_registry)

    if run_ctx.mode.value == "text":
        # TEXT mode requires TextTransport with reference resolution
        resolver = ReferenceResolver(
            state_hub=run_ctx.state_hub,
            artifact_store=run_ctx.artifact_store,
            memory_service=run_ctx.memory_service,
        )
        text_transport = TextTransport(
            text_counter=TextCounter(),
            resolver=resolver,
        )
        orchestrator = Orchestrator(router, text_transport=text_transport)
    else:
        # STRUCTURED mode uses direct routing
        orchestrator = Orchestrator(router)

    # Execute task with timing
    timer = TaskTimer()
    timer.start()

    try:
        final_envelope = orchestrator.run_task(task=task, ctx=run_ctx)

        # Extract answer from successful result
        if final_envelope.result is None or type(final_envelope.result) is not dict:
            raise ValueError("final envelope must have dict result")

        answer = final_envelope.result.get("answer")
        if type(answer) is not str:
            raise ValueError("final result must have str result['answer']")

        # Stop timer and record success
        latency_ms = timer.stop()
        run_ctx.metrics.set_latency_ms(latency_ms)
        run_ctx.metrics.set_success(True)

        # Build successful RunResult
        run_result = RunResult(
            task_id=run_ctx.task_id,
            success=True,
            answer=answer,
            error=None,
            metrics=run_ctx.metrics.snapshot().model_dump(mode="json"),
            trace_path=str(run_ctx.trace_logger.path),
        )

    except Exception as exc:
        # Capture runtime failures as structured error
        # Note: Configuration errors (TypeError, ValueError from validation)
        # and handshake failures are allowed to propagate as they indicate
        # setup problems, not task execution failures
        latency_ms = timer.stop()
        run_ctx.metrics.set_latency_ms(latency_ms)
        run_ctx.metrics.set_success(False)

        run_result = RunResult(
            task_id=run_ctx.task_id,
            success=False,
            answer="",
            error={
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            metrics=run_ctx.metrics.snapshot().model_dump(mode="json"),
            trace_path=str(run_ctx.trace_logger.path),
        )

    # Build and return raw record
    return RawRunRecord(
        experiment=experiment,
        task=task,
        task_hash=task_hash,
        seed=run_ctx.config.random_seed,
        use_sandbox=run_ctx.use_sandbox,
        run_result=run_result,
    )


def run_repeated(
    *,
    experiment: ExperimentConfig,
    task: str,
    seeds: list[int],
    run_factory: Callable[[int, int], tuple[RunContext, AgentRegistry]],
) -> list[RawRunRecord]:
    """Execute one experiment configuration multiple times with different seeds.

    This function orchestrates repeated execution of run_single() with fresh
    runtime contexts for each seed. The run_factory is responsible for creating
    independent runtime environments to ensure experiment isolation.

    Args:
        experiment: The A/B/C/D experiment configuration to apply
        task: The task description (non-empty string)
        seeds: List of random seeds (non-empty, each seed >= 0)
        run_factory: Callable that receives (run_index, seed) and returns
                     (fresh RunContext, fresh AgentRegistry)

    Returns:
        List of RawRunRecord, one per seed, in the same order as seeds

    Raises:
        TypeError: If arguments have wrong types
        ValueError: If seeds is empty, contains invalid values, or factory
                    returns ctx with mismatched seed
    """
    # Validate experiment
    if not isinstance(experiment, ExperimentConfig):
        raise TypeError("experiment must be an ExperimentConfig")

    # Validate task
    if type(task) is not str:
        raise TypeError("task must be a str")
    if task == "":
        raise ValueError("task must be a non-empty str")

    # Validate seeds list
    if type(seeds) is not list:
        raise TypeError("seeds must be a list[int]")
    if len(seeds) == 0:
        raise ValueError("seeds must be a non-empty list[int]")

    # Validate each seed before any execution
    for i, seed in enumerate(seeds):
        # Reject bool (bool is int subclass but not allowed)
        if type(seed) is bool:
            raise TypeError(f"seeds[{i}] must be int, not bool")
        if type(seed) is not int:
            raise TypeError(f"seeds[{i}] must be int, got {type(seed).__name__}")
        if seed < 0:
            raise ValueError(f"seeds[{i}] must be >= 0, got {seed}")

    # Validate run_factory
    if not callable(run_factory):
        raise TypeError("run_factory must be callable")

    # Execute repeated runs
    records: list[RawRunRecord] = []

    for run_index, seed in enumerate(seeds):
        # Call factory to get fresh runtime components
        factory_result = run_factory(run_index, seed)

        # Validate factory return type
        if type(factory_result) is not tuple:
            raise TypeError(
                f"run_factory must return tuple, got {type(factory_result).__name__}"
            )
        if len(factory_result) != 2:
            raise TypeError(
                f"run_factory must return tuple of length 2, got {len(factory_result)}"
            )

        ctx, agent_registry = factory_result

        # Validate returned types
        if not isinstance(ctx, RunContext):
            raise TypeError("run_factory must return (RunContext, AgentRegistry)")
        if not isinstance(agent_registry, AgentRegistry):
            raise TypeError("run_factory must return (RunContext, AgentRegistry)")

        # Verify seed matches
        if ctx.config.random_seed != seed:
            raise ValueError(
                f"run_factory returned ctx with random_seed={ctx.config.random_seed}, "
                f"expected {seed}"
            )

        # Execute single run
        record = run_single(
            experiment=experiment,
            task=task,
            ctx=ctx,
            agent_registry=agent_registry,
        )

        records.append(record)

    return records
