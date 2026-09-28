from agentipc.protocol.capability import AgentCapability
from agentipc.protocol.registry import CapabilityRegistry
from agentipc.runtime.agent_registry import AgentRegistry


_REQUIRED_AGENT_IDS = (
    "planner",
    "retriever",
    "executor",
    "summarizer",
)


def bootstrap_capabilities(
    *,
    agent_registry: AgentRegistry,
    capability_registry: CapabilityRegistry,
) -> None:
    if not isinstance(agent_registry, AgentRegistry):
        raise TypeError("agent_registry must be an AgentRegistry")
    if not isinstance(capability_registry, CapabilityRegistry):
        raise TypeError("capability_registry must be a CapabilityRegistry")

    agents = [
        agent_registry.get(agent_id)
        for agent_id in _REQUIRED_AGENT_IDS
    ]

    for agent in agents:
        capability_registry.register(
            AgentCapability(
                agent_id=agent.agent_id,
                capabilities=list(agent.capabilities),
            )
        )
