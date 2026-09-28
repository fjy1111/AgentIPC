from pathlib import Path

import pytest

from agentipc.agents.base import BaseAgent
from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.config import AgentIPCConfig
from agentipc.evaluation.trace import TraceEvent, TraceLogger
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.providers.factory import ProviderBundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.bootstrap import run_handshake
from agentipc.runtime.context import RunContext, RunMode
from agentipc.runtime.orchestrator import Orchestrator
from agentipc.runtime.router import Router


TASK = "diagnose openEuler network connectivity"

KNOWLEDGE = [
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


class PoisonService:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"forbidden infrastructure access: {name}")


def _events(path: Path) -> list[TraceEvent]:
    if not path.exists():
        return []
    return [
        TraceEvent.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _runtime(tmp_path: Path, *, mode: RunMode = RunMode.STRUCTURED, **flags):
    agent_registry = AgentRegistry()
    for agent in [
        PlannerAgent(),
        RetrieverAgent(KNOWLEDGE),
        ExecutorAgent(),
        SummarizerAgent(),
    ]:
        agent_registry.register(agent)

    capability_registry = CapabilityRegistry()
    trace_path = tmp_path / "trace.jsonl"
    trace_logger = TraceLogger(trace_path)
    provider_bundle = ProviderBundle(
        llm=MockLLMProvider(
            keyword_responses={
                "network": "Network troubleshooting completed.",
            },
            default_text="mock response",
        ),
        embedding=HashEmbeddingProvider(),
    )
    poison = PoisonService()
    ctx = RunContext(
        trace_id="trace-structured-min",
        task_id="task-structured-min",
        mode=mode,
        config=AgentIPCConfig(),
        registry=capability_registry,
        state_hub=poison,
        artifact_store=poison,
        memory_service=poison,
        metrics=poison,
        trace_logger=trace_logger,
        provider_bundle=provider_bundle,
        use_state=flags.get("use_state", False),
        use_memory=flags.get("use_memory", False),
        use_sandbox=flags.get("use_sandbox", False),
    )
    return agent_registry, capability_registry, trace_logger, trace_path, ctx


def test_structured_minimal_chain_runs_four_real_agents_in_order(tmp_path: Path) -> None:
    agent_registry, capability_registry, trace_logger, trace_path, ctx = _runtime(
        tmp_path
    )
    handshake_messages = run_handshake(
        agent_registry=agent_registry,
        capability_registry=capability_registry,
        trace_logger=trace_logger,
        trace_id=ctx.trace_id,
        task_id=ctx.task_id,
    )
    assert len(handshake_messages) == 12

    final = Orchestrator(Router(agent_registry)).run_task(task=TASK, ctx=ctx)

    assert final.message_type is MessageType.RESULT
    assert final.action is ActionType.SUMMARIZE
    assert final.status is MessageStatus.OK
    assert final.capability == "summarize"
    assert final.result is not None
    assert final.result["answer"] == "Network troubleshooting completed."
    assert "network-manager" in final.result["evidence_summary"]

    candidate = final.result["memory_candidate"]
    execution = candidate["payload"]["execution"]
    assert execution["operation"] == "identity"
    assert "network-manager" in execution["output"]["retrieved_document_ids"]

    events = _events(trace_path)
    assert len(events) == 20
    assert sum(e.message_type == MessageType.HELLO.value for e in events) == 4
    assert sum(e.message_type == MessageType.REGISTER.value for e in events) == 4
    assert sum(e.message_type == MessageType.ACK.value for e in events) == 4

    business = events[-8:]
    assert [(e.message_type, e.action) for e in business] == [
        ("REQUEST", "PLAN"),
        ("RESULT", "PLAN"),
        ("REQUEST", "RETRIEVE"),
        ("RESULT", "RETRIEVE"),
        ("REQUEST", "EXECUTE"),
        ("RESULT", "EXECUTE"),
        ("REQUEST", "SUMMARIZE"),
        ("RESULT", "SUMMARIZE"),
    ]
    assert [
        e.action
        for e in business
        if e.message_type == MessageType.REQUEST.value
    ] == ["PLAN", "RETRIEVE", "EXECUTE", "SUMMARIZE"]
    assert [(e.sender, e.receiver) for e in business] == [
        ("runtime", "planner"),
        ("planner", "runtime"),
        ("runtime", "retriever"),
        ("retriever", "runtime"),
        ("runtime", "executor"),
        ("executor", "runtime"),
        ("runtime", "summarizer"),
        ("summarizer", "runtime"),
    ]
    assert all(e.trace_id == ctx.trace_id for e in business)
    assert all(e.task_id == ctx.task_id for e in business)
    assert [e.step_id for e in business] == [
        "step-plan",
        "step-plan",
        "step-retrieve",
        "step-retrieve",
        "step-execute",
        "step-execute",
        "step-summarize",
        "step-summarize",
    ]
    assert all(e.state_refs == [] for e in business)
    assert all(e.artifact_refs == [] for e in business)
    assert all(e.memory_refs == [] for e in business)


def test_text_mode_is_rejected_before_business_trace(tmp_path: Path) -> None:
    agent_registry, _, _, trace_path, ctx = _runtime(tmp_path, mode=RunMode.TEXT)
    with pytest.raises(ValueError):
        Orchestrator(Router(agent_registry)).run_task(task=TASK, ctx=ctx)
    assert _events(trace_path) == []


@pytest.mark.parametrize("flag", ["use_state", "use_memory", "use_sandbox"])
def test_unsupported_feature_flags_are_rejected_before_dispatch(
    tmp_path: Path,
    flag: str,
) -> None:
    agent_registry, _, _, trace_path, ctx = _runtime(tmp_path, **{flag: True})
    with pytest.raises(ValueError):
        Orchestrator(Router(agent_registry)).run_task(task=TASK, ctx=ctx)
    assert _events(trace_path) == []


@pytest.mark.parametrize(
    ("task", "error"),
    [
        ("", ValueError),
        (123, TypeError),
    ],
)
def test_invalid_task_is_rejected_before_dispatch(
    tmp_path: Path,
    task: object,
    error: type[Exception],
) -> None:
    agent_registry, _, _, trace_path, ctx = _runtime(tmp_path)
    with pytest.raises(error):
        Orchestrator(Router(agent_registry)).run_task(task=task, ctx=ctx)  # type: ignore[arg-type]
    assert _events(trace_path) == []


def test_constructor_requires_router() -> None:
    with pytest.raises(TypeError):
        Orchestrator(object())  # type: ignore[arg-type]


def test_run_task_requires_real_context() -> None:
    orchestrator = Orchestrator(Router(AgentRegistry()))
    with pytest.raises(TypeError):
        orchestrator.run_task(task="x", ctx=object())  # type: ignore[arg-type]


class FailingPlanner(BaseAgent):
    agent_id = "planner"
    capabilities = ["plan"]

    def handle(self, envelope, ctx):
        raise RuntimeError("planner exploded")


def test_agent_exception_propagates_unchanged(tmp_path: Path) -> None:
    registry = AgentRegistry()
    registry.register(FailingPlanner())
    _, _, _, _, ctx = _runtime(tmp_path)
    error = RuntimeError
    with pytest.raises(error, match="planner exploded"):
        Orchestrator(Router(registry)).run_task(task=TASK, ctx=ctx)
