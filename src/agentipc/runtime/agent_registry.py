from agentipc.agents.base import BaseAgent


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, BaseAgent] = {}

    def register(self, agent: BaseAgent) -> None:
        if not isinstance(agent, BaseAgent):
            raise TypeError("agent must be a BaseAgent")

        agent_id = agent.agent_id
        self._validate_agent_id(agent_id)

        if agent_id in self._agents:
            raise ValueError(f"agent_id is already registered: {agent_id!r}")

        self._agents[agent_id] = agent

    def get(self, agent_id: str) -> BaseAgent:
        self._validate_agent_id(agent_id)
        return self._agents[agent_id]

    @staticmethod
    def _validate_agent_id(agent_id: object) -> None:
        if type(agent_id) is not str:
            raise TypeError("agent_id must be a str")
        if agent_id == "":
            raise ValueError("agent_id must be a non-empty str")
