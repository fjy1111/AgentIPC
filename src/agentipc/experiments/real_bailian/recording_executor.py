from __future__ import annotations

from typing import Any

from agentipc.agents.base import BaseAgent
from agentipc.agents.executor import ExecutorAgent
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.runtime.context import RunContext


class RecordingExecutorAgent(BaseAgent):
    """Thin wrapper that records Executor outputs without changing behavior."""

    agent_id = "executor"
    capabilities = ["execute"]

    def __init__(self, delegate: ExecutorAgent | None = None) -> None:
        self._delegate = ExecutorAgent() if delegate is None else delegate
        if not isinstance(self._delegate, ExecutorAgent):
            raise TypeError("delegate must be an ExecutorAgent")
        self.executions: list[dict[str, Any]] = []

    def handle(self, envelope: AgentEnvelope, ctx: RunContext) -> AgentEnvelope:
        response = self._delegate.handle(envelope, ctx)
        if response.result is not None:
            execution = response.result.get("execution")
            if isinstance(execution, dict):
                self.executions.append(execution)
        return response
