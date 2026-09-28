from agentipc.agents.executor import ExecutorAgent
from agentipc.agents.planner import PlannerAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.agents.summarizer import SummarizerAgent
from agentipc.protocol.capability import AgentCapability
from agentipc.protocol.enums import PROTOCOL_VERSION
from agentipc.protocol.registry import CapabilityRegistry


def test_agent_capabilities_are_exact_and_discoverable() -> None:
    assert PlannerAgent.agent_id == "planner"
    assert RetrieverAgent.agent_id == "retriever"
    assert ExecutorAgent.agent_id == "executor"
    assert SummarizerAgent.agent_id == "summarizer"

    assert PlannerAgent.capabilities == ["plan"]
    assert RetrieverAgent.capabilities == ["retrieve"]
    assert ExecutorAgent.capabilities == ["execute"]
    assert SummarizerAgent.capabilities == ["summarize"]

    agents = [
        PlannerAgent(),
        RetrieverAgent([]),
        ExecutorAgent(),
        SummarizerAgent(),
    ]
    registry = CapabilityRegistry()
    for agent in agents:
        registry.register(
            AgentCapability(
                agent_id=agent.agent_id,
                capabilities=list(agent.capabilities),
            )
        )

    expected = {
        "plan": "planner",
        "retrieve": "retriever",
        "execute": "executor",
        "summarize": "summarizer",
    }
    for capability, agent_id in expected.items():
        assert [
            item.agent_id for item in registry.discover(capability)
        ] == [agent_id]
        assert registry.supports(agent_id, capability) is True

        registered = registry.get(agent_id)
        assert registered is not None
        assert registered.protocol_versions == [PROTOCOL_VERSION]

    agent_to_capability = {agent_id: cap for cap, agent_id in expected.items()}
    for agent_id, own_capability in agent_to_capability.items():
        for capability in expected:
            assert registry.supports(agent_id, capability) is (
                capability == own_capability
            )

    assert registry.discover("unknown") == []
