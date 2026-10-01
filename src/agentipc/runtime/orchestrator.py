import math

from agentipc.artifacts.store import ArtifactStore
from agentipc.evaluation.metrics import MetricsCollector
from agentipc.evaluation.text_counter import TextCounter
from agentipc.memory.models import MemoryRecord, MemoryType
from agentipc.memory.service import MemoryService
from agentipc.protocol.codec import encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.refs import MemoryRef
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.router import Router
from agentipc.runtime.text_transport import TextTransport
from agentipc.state.hub import StateHub
from agentipc.state.plan_vector import PLAN_VECTOR_KIND, encode_plan_vector

_RUNTIME_ID = "runtime"


class _StateHubConfigurationError(TypeError, ValueError):
    pass


class _MemoryServiceConfigurationError(TypeError, ValueError):
    pass


def _is_reusable_codeact_execution(
    cached_execution: dict[str, object],
) -> bool:
    if cached_execution.get("operation") != "codeact":
        return False

    output = cached_execution.get("output")
    if type(output) is not dict:
        return False

    exit_code = output.get("exit_code")
    if type(exit_code) is not int or exit_code != 0:
        return False
    if type(output.get("stdout")) is not str:
        return False
    if type(output.get("stderr")) is not str:
        return False
    if output.get("timed_out") is not False:
        return False

    duration_ms = output.get("duration_ms")
    if type(duration_ms) not in (int, float):
        return False
    if type(duration_ms) is float and not math.isfinite(duration_ms):
        return False
    if duration_ms < 0:
        return False

    return True


def _select_reusable_memory(
    *,
    task: str,
    memory_refs: list[MemoryRef],
    memory_service: MemoryService,
) -> tuple[MemoryRef, MemoryRecord] | None:
    for ref in memory_refs:
        record = memory_service.get(ref.memory_id)
        if record is None:
            raise ValueError(
                "MemoryService.retrieve() returned a MemoryRef for a missing record: "
                f"{ref.memory_id!r}"
            )

        if record.task_topic != task:
            continue
        if record.memory_type is not MemoryType.RESULT:
            continue

        historical_answer = record.payload.get("answer")
        if type(historical_answer) is not str or historical_answer == "":
            continue

        cached_execution = record.payload.get("execution")
        if type(cached_execution) is not dict:
            continue

        operation = cached_execution.get("operation")
        if operation == "identity":
            if "output" not in cached_execution:
                continue
        elif operation == "codeact":
            if not _is_reusable_codeact_execution(cached_execution):
                continue
        else:
            continue

        return ref, record

    return None


class Orchestrator:
    def __init__(
        self,
        router: Router,
        *,
        text_transport: TextTransport | None = None,
        wire_counter: TextCounter | None = None,
        memory_fast_path: bool = False,
    ) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        if text_transport is not None and not isinstance(
            text_transport,
            TextTransport,
        ):
            raise TypeError("text_transport must be a TextTransport or None")
        if wire_counter is None:
            wire_counter = TextCounter()
        elif not isinstance(wire_counter, TextCounter):
            raise TypeError("wire_counter must be a TextCounter or None")
        if type(memory_fast_path) is not bool:
            raise TypeError("memory_fast_path must be a bool")

        self._router = router
        self._text_transport = text_transport
        self._wire_counter = wire_counter
        self._memory_fast_path = memory_fast_path

    def run_task(
        self,
        *,
        task: str,
        ctx: RunContext,
    ) -> AgentEnvelope:
        if not isinstance(ctx, RunContext):
            raise TypeError("ctx must be a RunContext")
        if type(task) is not str:
            raise TypeError("task must be a str")
        if task == "":
            raise ValueError("task must be a non-empty str")
        if type(ctx.use_state) is not bool:
            raise TypeError("ctx.use_state must be a bool")
        if type(ctx.use_memory) is not bool:
            raise TypeError("ctx.use_memory must be a bool")
        if type(ctx.use_sandbox) is not bool:
            raise TypeError("ctx.use_sandbox must be a bool")
        if ctx.use_memory and not isinstance(ctx.memory_service, MemoryService):
            raise _MemoryServiceConfigurationError(
                "use_memory=True requires ctx.memory_service to be a MemoryService"
            )
        if ctx.mode is RunMode.TEXT:
            if self._text_transport is None:
                raise ValueError("text mode requires TextTransport")
            if ctx.use_state:
                raise ValueError("text mode does not support use_state=True")
            if ctx.use_memory:
                raise ValueError("text mode does not support use_memory=True")
        elif ctx.mode is not RunMode.STRUCTURED:
            raise ValueError("unsupported run mode")
        if ctx.use_state and not isinstance(ctx.state_hub, StateHub):
            raise _StateHubConfigurationError(
                "use_state=True requires ctx.state_hub to be a StateHub"
            )

        if self._memory_fast_path and ctx.use_memory:
            fast_result = self._try_memory_fast_path(task=task, ctx=ctx)
            if fast_result is not None:
                return fast_result

        plan_state_ref = None
        try:
            planner_request = AgentEnvelope(
                trace_id=ctx.trace_id,
                task_id=ctx.task_id,
                step_id="step-plan",
                sender=_RUNTIME_ID,
                receiver="planner",
                message_type=MessageType.REQUEST,
                action=ActionType.PLAN,
                args={"task": task},
                state_refs=[],
                artifact_refs=[],
                memory_refs=[],
            )
            planner_result = self._dispatch(planner_request, ctx)
            planner_payload = self._require_stage_result(
                planner_result,
                planner_request,
                action=ActionType.PLAN,
                capability="plan",
                sender="planner",
            )
            plan = planner_payload.get("plan")
            if type(plan) is not dict:
                raise ValueError("planner result requires dict result['plan']")

            memory_refs: list[MemoryRef] = []
            selected_memory: tuple[MemoryRef, MemoryRecord] | None = None
            if ctx.use_memory:
                retrieved = ctx.memory_service.retrieve(task)
                if type(retrieved) is not list or not all(
                    isinstance(ref, MemoryRef) for ref in retrieved
                ):
                    raise ValueError(
                        "MemoryService.retrieve() must return a list[MemoryRef]"
                    )
                memory_refs = retrieved
                if isinstance(ctx.metrics, MetricsCollector):
                    ctx.metrics.increment(
                        "memory_retrieved",
                        len(memory_refs),
                    )
                selected_memory = _select_reusable_memory(
                    task=task,
                    memory_refs=memory_refs,
                    memory_service=ctx.memory_service,
                )

            if ctx.use_state:
                plan_vector = encode_plan_vector(plan)
                plan_state_ref = ctx.state_hub.put_array(
                    plan_vector,
                    kind=PLAN_VECTOR_KIND,
                    summary="planner plan vector",
                )

            retriever_request = AgentEnvelope(
                trace_id=ctx.trace_id,
                task_id=ctx.task_id,
                step_id="step-retrieve",
                sender=_RUNTIME_ID,
                receiver="retriever",
                message_type=MessageType.REQUEST,
                action=ActionType.RETRIEVE,
                args={"plan": plan},
                state_refs=[] if plan_state_ref is None else [plan_state_ref],
                artifact_refs=[],
                memory_refs=memory_refs,
            )
            retriever_result = self._dispatch(retriever_request, ctx)
            retriever_payload = self._require_stage_result(
                retriever_result,
                retriever_request,
                action=ActionType.RETRIEVE,
                capability="retrieve",
                sender="retriever",
            )
            evidence = retriever_payload.get("evidence")
            if type(evidence) is not list:
                raise ValueError("retriever result requires list result['evidence']")

            evidence_ref = None
            resolved_evidence = evidence
            if isinstance(ctx.artifact_store, ArtifactStore):
                evidence_ref = ctx.artifact_store.put_json(
                    evidence,
                    summary="retriever evidence",
                )
                resolved_evidence = ctx.artifact_store.get_json(evidence_ref)
                if type(resolved_evidence) is not list:
                    raise ValueError("resolved artifact evidence must be a list")
                if resolved_evidence != evidence:
                    raise ValueError(
                        "resolved artifact evidence does not match retriever evidence"
                    )

            if selected_memory is not None:
                _, selected_record = selected_memory
                cached_execution = selected_record.payload["execution"]
                operation = {
                    "name": "identity",
                    "value": cached_execution["output"],
                }
                if isinstance(ctx.metrics, MetricsCollector):
                    ctx.metrics.increment("memory_used")
            elif ctx.use_sandbox:
                operation = {
                    "name": "codeact",
                    "code": task,
                    "timeout_sec": 2.0,
                }
            else:
                operation = {
                    "name": "identity",
                    "value": {
                        "retrieved_document_ids": [
                            item["document_id"]
                            for item in resolved_evidence
                        ],
                    },
                }

            executor_request = AgentEnvelope(
                trace_id=ctx.trace_id,
                task_id=ctx.task_id,
                step_id="step-execute",
                sender=_RUNTIME_ID,
                receiver="executor",
                message_type=MessageType.REQUEST,
                action=ActionType.EXECUTE,
                args={"operation": operation},
                state_refs=[],
                artifact_refs=[] if evidence_ref is None else [evidence_ref],
                memory_refs=[],
            )
            executor_result = self._dispatch(executor_request, ctx)
            executor_payload = self._require_stage_result(
                executor_result,
                executor_request,
                action=ActionType.EXECUTE,
                capability="execute",
                sender="executor",
            )
            execution = executor_payload.get("execution")
            if type(execution) is not dict:
                raise ValueError("executor result requires dict result['execution']")

            summarizer_request = AgentEnvelope(
                trace_id=ctx.trace_id,
                task_id=ctx.task_id,
                step_id="step-summarize",
                sender=_RUNTIME_ID,
                receiver="summarizer",
                message_type=MessageType.REQUEST,
                action=ActionType.SUMMARIZE,
                args={
                    "task": task,
                    "evidence": resolved_evidence,
                    "execution": execution,
                },
                state_refs=[],
                artifact_refs=[],
                memory_refs=[],
            )
            summarizer_result = self._dispatch(summarizer_request, ctx)
            summarizer_payload = self._require_stage_result(
                summarizer_result,
                summarizer_request,
                action=ActionType.SUMMARIZE,
                capability="summarize",
                sender="summarizer",
            )

            if ctx.use_memory:
                candidate = summarizer_payload.get("memory_candidate")
                if type(candidate) is not dict:
                    raise ValueError(
                        "summarizer result requires dict result['memory_candidate']"
                    )

                record = MemoryRecord(
                    memory_id=f"mem_{ctx.task_id}",
                    **candidate,
                )
                stored = ctx.memory_service.write(record)
                if not isinstance(stored, MemoryRecord):
                    raise ValueError(
                        "MemoryService.write() must return a MemoryRecord"
                    )
                if stored.memory_id != record.memory_id:
                    raise ValueError(
                        "stored memory_id does not match requested memory_id"
                    )

                if selected_memory is not None:
                    final_answer = summarizer_payload.get("answer")
                    if type(final_answer) is not str or final_answer == "":
                        raise ValueError(
                            "summarizer result requires non-empty str result['answer']"
                        )

                    selected_ref, selected_record = selected_memory
                    historical_answer = selected_record.payload["answer"]
                    effective = final_answer == historical_answer
                    ctx.memory_service.mark_used(
                        selected_ref.memory_id,
                        effective=effective,
                    )
                    if isinstance(ctx.metrics, MetricsCollector):
                        if effective:
                            ctx.metrics.increment("memory_effective")
                        else:
                            ctx.metrics.increment("memory_harmful")

            return summarizer_result
        finally:
            if plan_state_ref is not None:
                ctx.state_hub.release(plan_state_ref)

    def _try_memory_fast_path(
        self,
        *,
        task: str,
        ctx: RunContext,
    ) -> AgentEnvelope | None:
        record = ctx.memory_service.get_exact_validated_result(task)
        if record is None:
            return None

        historical_answer = record.payload.get("answer")
        if type(historical_answer) is not str or historical_answer == "":
            return None

        cached_execution = record.payload.get("execution")
        if type(cached_execution) is not dict:
            return None
        operation = cached_execution.get("operation")
        if operation == "identity":
            if "output" not in cached_execution:
                return None
        elif operation == "codeact":
            if not _is_reusable_codeact_execution(cached_execution):
                return None
        else:
            return None

        ctx.memory_service.mark_used(record.memory_id, effective=True)
        if isinstance(ctx.metrics, MetricsCollector):
            ctx.metrics.increment("memory_retrieved")
            ctx.metrics.increment("memory_used")
            ctx.metrics.increment("memory_effective")
            ctx.metrics.increment("fast_path_hit_count")

        evidence_summary = record.payload.get("evidence_summary")
        if type(evidence_summary) is not str:
            evidence_summary = "Reused exact evaluator-validated RESULT memory."

        return AgentEnvelope(
            trace_id=ctx.trace_id,
            task_id=ctx.task_id,
            step_id="step-memory-fast-path",
            sender=_RUNTIME_ID,
            receiver=_RUNTIME_ID,
            message_type=MessageType.RESULT,
            action=ActionType.SUMMARIZE,
            capability="memory_fast_path",
            result={
                "answer": historical_answer,
                "evidence_summary": evidence_summary,
                "fast_path": True,
            },
            status=MessageStatus.OK,
            state_refs=[],
            artifact_refs=[],
            memory_refs=[
                MemoryRef(
                    memory_id=record.memory_id,
                    score=1.0,
                    match_type="exact_validated",
                    summary=record.summary,
                )
            ],
        )

    def _dispatch(
        self,
        request: AgentEnvelope,
        ctx: RunContext,
    ) -> AgentEnvelope:
        if ctx.mode is RunMode.STRUCTURED:
            return self._dispatch_structured(request, ctx)

        if ctx.mode is RunMode.TEXT:
            assert self._text_transport is not None
            return self._text_transport.dispatch(
                request,
                ctx=ctx,
                router=self._router,
            )

        raise ValueError("unsupported run mode")

    def _dispatch_structured(
        self,
        request: AgentEnvelope,
        ctx: RunContext,
    ) -> AgentEnvelope:
        self._record_structured_envelope(request, ctx)
        response = self._router.dispatch(request, ctx)
        if not isinstance(response, AgentEnvelope):
            raise ValueError("stage response must be an AgentEnvelope")
        self._record_structured_envelope(response, ctx)
        return response

    def _record_structured_envelope(
        self,
        envelope: AgentEnvelope,
        ctx: RunContext,
    ) -> None:
        if isinstance(ctx.metrics, MetricsCollector):
            payload = encode(envelope)
            wire_text = payload.decode("utf-8")
            count = self._wire_counter.count(wire_text)
            ctx.metrics.increment("message_count")
            ctx.metrics.increment("wire_chars", count.text_chars)
            ctx.metrics.increment("wire_tokens", count.text_tokens)
            ctx.metrics.increment("wire_bytes", len(payload))
            ctx.metrics.increment("protocol_bytes", len(payload))
            if envelope.state_refs:
                ctx.metrics.increment(
                    "state_transfer_count",
                    len(envelope.state_refs),
                )
                ctx.metrics.increment(
                    "state_bytes",
                    sum(ref.nbytes for ref in envelope.state_refs),
                )
            if envelope.artifact_refs:
                ctx.metrics.increment(
                    "artifact_ref_count",
                    len(envelope.artifact_refs),
                )
                ctx.metrics.increment(
                    "artifact_payload_bytes",
                    sum(ref.size_bytes for ref in envelope.artifact_refs),
                )

        ctx.trace_logger.log_envelope(envelope)

    @staticmethod
    def _require_stage_result(
        response: AgentEnvelope,
        request: AgentEnvelope,
        *,
        action: ActionType,
        capability: str,
        sender: str,
    ) -> dict[str, object]:
        if not isinstance(response, AgentEnvelope):
            raise ValueError("stage response must be an AgentEnvelope")
        if response.message_type is not MessageType.RESULT:
            raise ValueError("stage response must be a RESULT envelope")
        if response.action is not action:
            raise ValueError("stage response action does not match request")
        if response.status is not MessageStatus.OK:
            raise ValueError("stage response status must be OK")
        if response.capability != capability:
            raise ValueError("stage response capability does not match stage")
        if response.sender != sender:
            raise ValueError("stage response sender does not match stage")
        if response.receiver != _RUNTIME_ID:
            raise ValueError("stage response receiver must be runtime")
        if response.trace_id != request.trace_id:
            raise ValueError("stage response trace_id does not match request")
        if response.task_id != request.task_id:
            raise ValueError("stage response task_id does not match request")
        if response.step_id != request.step_id:
            raise ValueError("stage response step_id does not match request")
        if type(response.result) is not dict:
            raise ValueError("stage response result must be a dict")
        return response.result