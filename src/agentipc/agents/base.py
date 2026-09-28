from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from agentipc.protocol.envelope import AgentEnvelope

if TYPE_CHECKING:
    from agentipc.runtime.context import RunContext


class BaseAgent(ABC):
    agent_id: str
    capabilities: list[str]

    @abstractmethod
    def handle(
        self,
        envelope: AgentEnvelope,
        ctx: "RunContext",
    ) -> AgentEnvelope:
        ...
