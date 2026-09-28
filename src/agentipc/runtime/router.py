from agentipc.protocol.envelope import AgentEnvelope
from agentipc.runtime.agent_registry import AgentRegistry
from agentipc.runtime.context import RunContext


class Router:
    def __init__(self, agent_registry: AgentRegistry) -> None:
        if not isinstance(agent_registry, AgentRegistry):
            raise TypeError("agent_registry must be an AgentRegistry")
        self._agent_registry = agent_registry

    def dispatch(
        self,
        envelope: AgentEnvelope,
        ctx: RunContext,
    ) -> AgentEnvelope:
        if not isinstance(envelope, AgentEnvelope):
            raise TypeError("envelope must be an AgentEnvelope")

        agent = self._agent_registry.get(envelope.receiver)
        return agent.handle(envelope, ctx)
