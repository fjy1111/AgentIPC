from __future__ import annotations

from typing import TYPE_CHECKING

from agentipc.agents.base import BaseAgent
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType

if TYPE_CHECKING:
    from agentipc.runtime.context import RunContext


class PlannerAgent(BaseAgent):
    agent_id = "planner"
    capabilities: list[str] = []

    def handle(
        self,
        envelope: AgentEnvelope,
        ctx: "RunContext",
    ) -> AgentEnvelope:
        if envelope.message_type is not MessageType.REQUEST:
            raise ValueError("planner only accepts REQUEST envelopes")
        if envelope.action is not ActionType.PLAN:
            raise ValueError("planner only accepts PLAN actions")

        if "task" not in envelope.args:
            raise ValueError("planner requires args['task']")
        task = envelope.args["task"]
        if type(task) is not str:
            raise TypeError("task must be a str")
        if task == "":
            raise ValueError("task must be non-empty")

        if "retrieval_topics" in envelope.args:
            raw_topics = envelope.args["retrieval_topics"]
            if type(raw_topics) is not list:
                raise TypeError("retrieval_topics must be a list[str]")
            retrieval_topics: list[str] = []
            for topic in raw_topics:
                if type(topic) is not str:
                    raise TypeError("retrieval_topics items must be str")
                if topic == "":
                    raise ValueError("retrieval_topics items must be non-empty")
                retrieval_topics.append(topic)
        else:
            retrieval_topics = [task]

        if "requires_execution" in envelope.args:
            requires_execution = envelope.args["requires_execution"]
            if type(requires_execution) is not bool:
                raise TypeError("requires_execution must be a bool")
        else:
            requires_execution = False

        messages = [
            {
                "role": "system",
                "content": (
                    "Produce a concise planning note for the AgentIPC planner. "
                    "Do not return protocol envelopes."
                ),
            },
            {
                "role": "user",
                "content": task,
            },
        ]
        llm_response = ctx.provider_bundle.llm.complete(
            messages,
            temperature=0.0,
        )

        plan = {
            "task": task,
            "steps": [
                "retrieve",
                "execute",
                "summarize",
            ],
            "required_capabilities": [
                "retrieve",
                "execute",
                "summarize",
            ],
            "retrieval_topics": retrieval_topics,
            "requires_execution": requires_execution,
            "planner_note": llm_response.text,
        }

        return AgentEnvelope(
            trace_id=envelope.trace_id,
            task_id=envelope.task_id,
            step_id=envelope.step_id,
            sender=self.agent_id,
            receiver=envelope.sender,
            message_type=MessageType.RESULT,
            action=ActionType.PLAN,
            capability="plan",
            result={"plan": plan},
            status=MessageStatus.OK,
        )
