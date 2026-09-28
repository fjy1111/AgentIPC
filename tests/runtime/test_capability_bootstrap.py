import pytest

from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.protocol.enums import PROTOCOL_VERSION
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.bootstrap import bootstrap_capabilities


def _agents():
    return [PlannerAgent(), RetrieverAgent([]), ExecutorAgent(), SummarizerAgent()]


def _registry_with_all_agents():
    registry = AgentRegistry()
    agents = _agents()
    for agent in agents:
        registry.register(agent)
    return registry, agents


def _ids(records):
    return [record.agent_id for record in records]


def test_bootstrap_registers_discoverable_capabilities_and_copies_lists() -> None:
    agent_registry, agents = _registry_with_all_agents()
    capability_registry = CapabilityRegistry()

    result = bootstrap_capabilities(
        agent_registry=agent_registry,
        capability_registry=capability_registry,
    )

    assert result is None
    assert _ids(capability_registry.discover("plan")) == ["planner"]
    assert _ids(capability_registry.discover("retrieve")) == ["retriever"]
    assert _ids(capability_registry.discover("execute")) == ["executor"]
    assert _ids(capability_registry.discover("summarize")) == ["summarizer"]

    assert capability_registry.supports("planner", "plan") is True
    assert capability_registry.supports("retriever", "retrieve") is True
    assert capability_registry.supports("executor", "execute") is True
    assert capability_registry.supports("summarizer", "summarize") is True
    assert capability_registry.supports("planner", "execute") is False
    assert capability_registry.supports("executor", "plan") is False

    for agent in agents:
        record = capability_registry.get(agent.agent_id)
        assert record is not None
        assert record.agent_id == agent.agent_id
        assert record.capabilities == agent.capabilities
        assert record.capabilities is not agent.capabilities
        assert record.protocol_versions == [PROTOCOL_VERSION]
        assert record.metadata == {}


def test_missing_agent_fails_before_any_capability_mutation() -> None:
    agent_registry = AgentRegistry()
    for agent in [PlannerAgent(), RetrieverAgent([]), ExecutorAgent()]:
        agent_registry.register(agent)
    capability_registry = CapabilityRegistry()

    with pytest.raises(KeyError):
        bootstrap_capabilities(
            agent_registry=agent_registry,
            capability_registry=capability_registry,
        )

    for agent_id in ["planner", "retriever", "executor", "summarizer"]:
        assert capability_registry.get(agent_id) is None


def test_empty_agent_registry_fails_without_mutation() -> None:
    capability_registry = CapabilityRegistry()
    with pytest.raises(KeyError):
        bootstrap_capabilities(
            agent_registry=AgentRegistry(),
            capability_registry=capability_registry,
        )
    assert capability_registry.get("planner") is None


def test_rebootstrap_is_idempotent_under_registry_replace_semantics() -> None:
    agent_registry, _ = _registry_with_all_agents()
    capability_registry = CapabilityRegistry()

    bootstrap_capabilities(agent_registry=agent_registry, capability_registry=capability_registry)
    bootstrap_capabilities(agent_registry=agent_registry, capability_registry=capability_registry)

    assert _ids(capability_registry.discover("plan")) == ["planner"]
    assert _ids(capability_registry.discover("retrieve")) == ["retriever"]
    assert _ids(capability_registry.discover("execute")) == ["executor"]
    assert _ids(capability_registry.discover("summarize")) == ["summarizer"]


def test_invalid_agent_registry_is_rejected() -> None:
    with pytest.raises(TypeError):
        bootstrap_capabilities(
            agent_registry=object(),  # type: ignore[arg-type]
            capability_registry=CapabilityRegistry(),
        )


def test_invalid_capability_registry_is_rejected() -> None:
    with pytest.raises(TypeError):
        bootstrap_capabilities(
            agent_registry=AgentRegistry(),
            capability_registry=object(),  # type: ignore[arg-type]
        )
