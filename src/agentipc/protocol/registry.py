from agentipc.protocol.capability import AgentCapability


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[str, AgentCapability] = {}

    def register(self, capability: AgentCapability) -> None:
        self._capabilities[capability.agent_id] = capability

    def get(self, agent_id: str) -> AgentCapability | None:
        return self._capabilities.get(agent_id)

    def discover(self, required: str) -> list[AgentCapability]:
        matches = [
            capability
            for capability in self._capabilities.values()
            if required in capability.capabilities
        ]
        return sorted(matches, key=lambda capability: capability.agent_id)

    def supports(self, agent_id: str, capability: str) -> bool:
        registered = self.get(agent_id)
        if registered is None:
            return False
        return capability in registered.capabilities