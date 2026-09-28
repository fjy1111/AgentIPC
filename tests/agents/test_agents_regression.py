from types import SimpleNamespace

from agentipc.agents.base import BaseAgent
from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.memory.models import MemoryRecord
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider


TRACE_ID = "trace-agent-regression"
TASK_ID = "task-agent-regression"
TASK = "diagnose openEuler network connectivity"


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected infrastructure access: {name}")


def _assert_result(
    request: AgentEnvelope,
    result: AgentEnvelope,
    *,
    action: ActionType,
    capability: str,
) -> None:
    assert result.message_type is MessageType.RESULT
    assert result.action is action
    assert result.capability == capability
    assert result.status is MessageStatus.OK
    assert result.trace_id == request.trace_id == TRACE_ID
    assert result.task_id == request.task_id == TASK_ID
    assert result.step_id == request.step_id
    assert decode(encode(result)) == result


def test_four_agent_package_regression() -> None:
    llm = MockLLMProvider(
        keyword_responses={
            "network": "Network troubleshooting completed.",
        },
        default_text="mock response",
    )
    ctx = SimpleNamespace(
        provider_bundle=SimpleNamespace(
            llm=llm,
            embedding=HashEmbeddingProvider(),
        ),
        state_hub=PoisonService(),
        memory_service=PoisonService(),
        artifact_store=PoisonService(),
        metrics=PoisonService(),
        trace_logger=PoisonService(),
    )
    knowledge = [
        {
            "document_id": "network-manager",
            "text": (
                "NetworkManager manages openEuler network "
                "connections and connectivity."
            ),
            "keywords": [
                "NetworkManager",
                "openEuler",
                "network connectivity",
            ],
        },
        {
            "document_id": "filesystem",
            "text": "Use fsck for filesystem diagnostics.",
            "keywords": [
                "filesystem",
                "fsck",
            ],
        },
    ]

    planner = PlannerAgent()
    retriever = RetrieverAgent(knowledge)
    executor = ExecutorAgent()
    summarizer = SummarizerAgent()
    agents = [planner, retriever, executor, summarizer]

    assert all(isinstance(agent, BaseAgent) for agent in agents)
    assert planner.capabilities == ["plan"]
    assert retriever.capabilities == ["retrieve"]
    assert executor.capabilities == ["execute"]
    assert summarizer.capabilities == ["summarize"]

    plan_request = AgentEnvelope(
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step-plan",
        sender="runtime-test",
        receiver="planner",
        message_type=MessageType.REQUEST,
        action=ActionType.PLAN,
        args={
            "task": TASK,
            "retrieval_topics": [
                "NetworkManager",
                "network connectivity",
            ],
            "requires_execution": False,
        },
    )
    plan_result = planner.handle(plan_request, ctx)
    _assert_result(
        plan_request,
        plan_result,
        action=ActionType.PLAN,
        capability="plan",
    )
    assert plan_result.result is not None
    plan = plan_result.result["plan"]
    assert isinstance(plan, dict)

    retrieve_request = AgentEnvelope(
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step-retrieve",
        sender="runtime-test",
        receiver="retriever",
        message_type=MessageType.REQUEST,
        action=ActionType.RETRIEVE,
        args={
            "plan": plan,
            "top_k": 1,
        },
        state_refs=[],
    )
    retrieve_result = retriever.handle(retrieve_request, ctx)
    _assert_result(
        retrieve_request,
        retrieve_result,
        action=ActionType.RETRIEVE,
        capability="retrieve",
    )
    assert retrieve_result.result is not None
    evidence = retrieve_result.result["evidence"]
    assert isinstance(evidence, list)
    assert evidence
    assert evidence[0]["document_id"] == "network-manager"

    execute_request = AgentEnvelope(
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step-execute",
        sender="runtime-test",
        receiver="executor",
        message_type=MessageType.REQUEST,
        action=ActionType.EXECUTE,
        args={
            "operation": {
                "name": "identity",
                "value": {
                    "retrieved_document_ids": [
                        item["document_id"] for item in evidence
                    ],
                },
            },
        },
    )
    execute_result = executor.handle(execute_request, ctx)
    _assert_result(
        execute_request,
        execute_result,
        action=ActionType.EXECUTE,
        capability="execute",
    )
    assert execute_result.result is not None
    execution = execute_result.result["execution"]
    assert isinstance(execution, dict)
    assert execution["operation"] == "identity"
    assert execution["output"]["retrieved_document_ids"] == [
        "network-manager"
    ]

    summarize_request = AgentEnvelope(
        trace_id=TRACE_ID,
        task_id=TASK_ID,
        step_id="step-summarize",
        sender="runtime-test",
        receiver="summarizer",
        message_type=MessageType.REQUEST,
        action=ActionType.SUMMARIZE,
        args={
            "task": TASK,
            "evidence": evidence,
            "execution": execution,
            "tags": ["network", "openEuler"],
            "keywords": ["NetworkManager", "connectivity"],
        },
    )
    summarize_result = summarizer.handle(summarize_request, ctx)
    _assert_result(
        summarize_request,
        summarize_result,
        action=ActionType.SUMMARIZE,
        capability="summarize",
    )
    assert summarize_result.result is not None
    assert summarize_result.result["answer"] == (
        "Network troubleshooting completed."
    )
    assert summarize_result.result["evidence_summary"]
    candidate = summarize_result.result["memory_candidate"]
    assert isinstance(candidate, dict)
    MemoryRecord(
        memory_id="mem-agent-regression",
        **candidate,
    )

    results = [
        plan_result,
        retrieve_result,
        execute_result,
        summarize_result,
    ]
    assert [result.action for result in results] == [
        ActionType.PLAN,
        ActionType.RETRIEVE,
        ActionType.EXECUTE,
        ActionType.SUMMARIZE,
    ]
    for agent, result in zip(agents, results, strict=True):
        assert result.capability == agent.capabilities[0]
