from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.router import Router

_RUNTIME_ID = "runtime"


class Orchestrator:
    def __init__(self, router: Router) -> None:
        if not isinstance(router, Router):
            raise TypeError("router must be a Router")
        self._router = router

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
        if ctx.mode is not RunMode.STRUCTURED:
            raise ValueError("T095 only supports structured mode")
        if ctx.use_state is not False:
            raise ValueError("T095 requires use_state=False")
        if ctx.use_memory is not False:
            raise ValueError("T095 requires use_memory=False")
        if ctx.use_sandbox is not False:
            raise ValueError("T095 requires use_sandbox=False")

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

        retriever_request = AgentEnvelope(
            trace_id=ctx.trace_id,
            task_id=ctx.task_id,
            step_id="step-retrieve",
            sender=_RUNTIME_ID,
            receiver="retriever",
            message_type=MessageType.REQUEST,
            action=ActionType.RETRIEVE,
            args={"plan": plan},
            state_refs=[],
            artifact_refs=[],
            memory_refs=[],
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

        operation = {
            "name": "identity",
            "value": {
                "retrieved_document_ids": [
                    item["document_id"]
                    for item in evidence
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
            artifact_refs=[],
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
                "evidence": evidence,
                "execution": execution,
            },
            state_refs=[],
            artifact_refs=[],
            memory_refs=[],
        )
        summarizer_result = self._dispatch(summarizer_request, ctx)
        self._require_stage_result(
            summarizer_result,
            summarizer_request,
            action=ActionType.SUMMARIZE,
            capability="summarize",
            sender="summarizer",
        )
        return summarizer_result

    def _dispatch(
        self,
        request: AgentEnvelope,
        ctx: RunContext,
    ) -> AgentEnvelope:
        ctx.trace_logger.log_envelope(request)
        response = self._router.dispatch(request, ctx)
        if not isinstance(response, AgentEnvelope):
            raise ValueError("stage response must be an AgentEnvelope")
        ctx.trace_logger.log_envelope(response)
        return response

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
