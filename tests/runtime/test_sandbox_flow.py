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
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router


TASK = 'print("agentipc-codeact-runtime")'
EMBEDDING_DIM = 64

KNOWLEDGE = [
    {
        "document_id": "codeact-runtime",
        "text": "AgentIPC CodeAct runtime executes Python tasks in a sandbox.",
        "keywords": ["agentipc", "codeact", "runtime", "python"],
    }
]


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"forbidden infrastructure access: {name}")


class RecordingRouter(Router):
    def __init__(self, agent_registry: AgentRegistry) -> None:
        super().__init__(agent_registry)
        self.requests: list[AgentEnvelope] = []
        self.responses: list[AgentEnvelope] = []

    def dispatch(
        self,
        envelope: AgentEnvelope,
        ctx: RunContext,
    ) -> AgentEnvelope:
        self.requests.append(envelope.model_copy(deep=True))
        response = super().dispatch(envelope, ctx)
        self.responses.append(response.model_copy(deep=True))
        return response


def _build_agents() -> AgentRegistry:
    registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent(),
        SummarizerAgent(),
    ]:
        registry.register(agent)
    return registry


def _build_provider_bundle() -> ProviderBundle:
    return ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={
                "agentipc-codeact-runtime": "CodeAct runtime completed.",
            },
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(dim=EMBEDDING_DIM),
    )


def _build_context(
    tmp_path: Path,
    *,
    name: str,
    metrics: MetricsCollector,
    use_sandbox: bool,
) -> RunContext:
    poison = PoisonService()
    return RunContext(
        trace_id=f"trace-{name}",
        task_id=f"task-{name}",
        mode=RunMode.STRUCTURED,
        config=AgentIPCConfig(),
        registry=CapabilityRegistry(),
        state_hub=poison,  # type: ignore[arg-type]
        artifact_store=poison,  # type: ignore[arg-type]
        memory_service=poison,  # type: ignore[arg-type]
        metrics=metrics,
        trace_logger=TraceLogger(tmp_path / f"{name}.jsonl"),
        provider_bundle=_build_provider_bundle(),
        use_state=False,
        use_memory=False,
        use_sandbox=use_sandbox,
    )


def _single_envelope(
    envelopes: list[AgentEnvelope],
    *,
    action: ActionType,
) -> AgentEnvelope:
    matches = [envelope for envelope in envelopes if envelope.action is action]
    assert len(matches) == 1
    return matches[0]


def test_structured_runtime_codeact_reaches_real_sandbox_and_summarizer(
    tmp_path: Path,
) -> None:
    metrics = MetricsCollector()
    ctx = _build_context(
        tmp_path,
        name="sandbox-flow",
        metrics=metrics,
        use_sandbox=True,
    )
    router = RecordingRouter(_build_agents())

    final = Orchestrator(router).run_task(task=TASK, ctx=ctx)

    executor_request = _single_envelope(
        router.requests,
        action=ActionType.EXECUTE,
    )
    assert executor_request.message_type is MessageType.REQUEST
    assert executor_request.args["operation"] == {
        "name": "codeact",
        "code": TASK,
        "timeout_sec": 2.0,
    }

    executor_result = _single_envelope(
        router.responses,
        action=ActionType.EXECUTE,
    )
    assert executor_result.message_type is MessageType.RESULT
    assert executor_result.status is MessageStatus.OK
    assert executor_result.capability == "execute"
    assert executor_result.result is not None

    execution = executor_result.result["execution"]
    assert execution["operation"] == "codeact"
    output = execution["output"]
    assert output["exit_code"] == 0
    assert output["stdout"] == "agentipc-codeact-runtime\n"
    assert output["stderr"] == ""
    assert output["timed_out"] is False
    assert output["duration_ms"] >= 0

    snapshot = metrics.snapshot()
    assert snapshot.message_count == 8
    assert snapshot.tool_call_count == 1
    assert snapshot.repeated_tool_call_count == 0
    assert snapshot.protocol_bytes > 0
    assert snapshot.state_transfer_count == 0
    assert snapshot.state_bytes == 0
    assert snapshot.memory_retrieved == 0
    assert snapshot.memory_used == 0
    assert snapshot.memory_effective == 0
    assert snapshot.memory_harmful == 0

    assert len(router.requests) == 4
    assert len(router.responses) == 4
    assert [envelope.action for envelope in router.requests] == [
        ActionType.PLAN,
        ActionType.RETRIEVE,
        ActionType.EXECUTE,
        ActionType.SUMMARIZE,
    ]

    assert isinstance(final, AgentEnvelope)
    assert final.message_type is MessageType.RESULT
    assert final.action is ActionType.SUMMARIZE
    assert final.status is MessageStatus.OK
    assert final.capability == "summarize"
    assert final.result is not None
    assert final.result["answer"] == "CodeAct runtime completed."


def test_structured_runtime_sandbox_false_preserves_identity_path(
    tmp_path: Path,
) -> None:
    metrics = MetricsCollector()
    ctx = _build_context(
        tmp_path,
        name="sandbox-disabled",
        metrics=metrics,
        use_sandbox=False,
    )
    router = RecordingRouter(_build_agents())

    final = Orchestrator(router).run_task(task=TASK, ctx=ctx)

    executor_request = _single_envelope(
        router.requests,
        action=ActionType.EXECUTE,
    )
    operation = executor_request.args["operation"]
    assert type(operation) is dict
    assert operation["name"] == "identity"
    assert "code" not in operation

    executor_result = _single_envelope(
        router.responses,
        action=ActionType.EXECUTE,
    )
    assert executor_result.result is not None
    execution = executor_result.result["execution"]
    assert execution["operation"] == "identity"

    snapshot = metrics.snapshot()
    assert snapshot.message_count == 8
    assert snapshot.tool_call_count == 0
    assert snapshot.repeated_tool_call_count == 0

    assert final.message_type is MessageType.RESULT
    assert final.action is ActionType.SUMMARIZE
    assert final.status is MessageStatus.OK


@pytest.mark.parametrize(
    "invalid",
    [None, 0, 1, "true", [], {}],
)
def test_orchestrator_requires_use_sandbox_to_be_exact_bool(
    tmp_path: Path,
    invalid: object,
) -> None:
    metrics = MetricsCollector()
    ctx = _build_context(
        tmp_path,
        name="invalid-sandbox",
        metrics=metrics,
        use_sandbox=False,
    )
    ctx.use_sandbox = invalid  # type: ignore[assignment]
    router = RecordingRouter(_build_agents())

    with pytest.raises(TypeError, match="use_sandbox"):
        Orchestrator(router).run_task(task=TASK, ctx=ctx)

    assert router.requests == []
    assert router.responses == []
    assert metrics.snapshot().message_count == 0
    assert metrics.snapshot().tool_call_count == 0