from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from agentipc.agents.base import BaseAgent
from agentipc.memory.models import MemoryType
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType

if TYPE_CHECKING:
    from agentipc.runtime.context import RunContext


_EVIDENCE_FIELDS = frozenset({"document_id", "text", "score"})
_SYSTEM_PROMPT = (
    "Produce a concise final answer for the AgentIPC task. "
    "Use only the supplied task, evidence summary, and execution result. "
    "Do not return protocol envelopes or hidden reasoning."
)


class SummarizerAgent(BaseAgent):
    agent_id = "summarizer"
    capabilities: list[str] = ["summarize"]

    def handle(
        self,
        envelope: AgentEnvelope,
        ctx: "RunContext",
    ) -> AgentEnvelope:
        if envelope.message_type is not MessageType.REQUEST:
            raise ValueError("summarizer only accepts REQUEST envelopes")
        if envelope.action is not ActionType.SUMMARIZE:
            raise ValueError("summarizer only accepts SUMMARIZE actions")

        if "task" not in envelope.args:
            raise ValueError("summarizer requires args['task']")
        task = envelope.args["task"]
        if type(task) is not str:
            raise TypeError("task must be a str")
        if task == "":
            raise ValueError("task must be non-empty")

        if "evidence" not in envelope.args:
            raise ValueError("summarizer requires args['evidence']")
        evidence = envelope.args["evidence"]
        _validate_evidence(evidence)

        if "execution" not in envelope.args:
            raise ValueError("summarizer requires args['execution']")
        execution = envelope.args["execution"]
        if type(execution) is not dict:
            raise TypeError("execution must be a dict")
        if "operation" not in execution:
            raise ValueError("execution requires operation")
        operation = execution["operation"]
        if type(operation) is not str:
            raise TypeError("execution operation must be a str")
        if operation == "":
            raise ValueError("execution operation must be non-empty")
        _validate_json_compatible(execution, field_name="execution")

        tags = _validate_string_list(
            envelope.args.get("tags", []),
            field_name="tags",
        )
        keywords = _validate_string_list(
            envelope.args.get("keywords", []),
            field_name="keywords",
        )

        evidence_summary = _build_evidence_summary(evidence)
        user_content = json.dumps(
            {
                "task": task,
                "evidence_summary": evidence_summary,
                "execution": execution,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        messages = [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        llm_response = ctx.provider_bundle.llm.complete(
            messages,
            temperature=0.0,
        )
        answer = llm_response.text
        if type(answer) is not str:
            raise TypeError("provider response text must be a str")
        if answer == "":
            raise ValueError("provider response text must be non-empty")

        memory_candidate = {
            "source_agent": self.agent_id,
            "task_topic": task,
            "summary": answer,
            "memory_type": MemoryType.RESULT.value,
            "tags": tags,
            "keywords": keywords,
            "payload": {
                "answer": answer,
                "evidence_summary": evidence_summary,
                "execution": execution,
            },
        }

        return AgentEnvelope(
            trace_id=envelope.trace_id,
            task_id=envelope.task_id,
            step_id=envelope.step_id,
            sender=self.agent_id,
            receiver=envelope.sender,
            message_type=MessageType.RESULT,
            action=ActionType.SUMMARIZE,
            capability="summarize",
            result={
                "answer": answer,
                "evidence_summary": evidence_summary,
                "memory_candidate": memory_candidate,
            },
            status=MessageStatus.OK,
            state_refs=[],
            artifact_refs=[],
            memory_refs=[],
        )


def _validate_evidence(evidence: object) -> None:
    if type(evidence) is not list:
        raise TypeError("evidence must be a list")

    for item in evidence:
        if type(item) is not dict:
            raise TypeError("evidence items must be dict")

        unknown = set(item) - _EVIDENCE_FIELDS
        if unknown:
            fields = ", ".join(sorted(repr(field) for field in unknown))
            raise ValueError(f"evidence item contains unknown fields: {fields}")

        missing = _EVIDENCE_FIELDS - set(item)
        if missing:
            fields = ", ".join(sorted(repr(field) for field in missing))
            raise ValueError(f"evidence item missing required fields: {fields}")

        document_id = item["document_id"]
        if type(document_id) is not str:
            raise TypeError("evidence document_id must be a str")
        if document_id == "":
            raise ValueError("evidence document_id must be non-empty")

        text = item["text"]
        if type(text) is not str:
            raise TypeError("evidence text must be a str")
        if text == "":
            raise ValueError("evidence text must be non-empty")

        score = item["score"]
        if type(score) not in (int, float):
            raise TypeError("evidence score must be an int or float")
        if type(score) is float and not math.isfinite(score):
            raise ValueError("evidence score must be finite")
        if score < 0:
            raise ValueError("evidence score must be >= 0")


def _validate_string_list(value: object, *, field_name: str) -> list[str]:
    if type(value) is not list:
        raise TypeError(f"{field_name} must be a list[str]")

    result: list[str] = []
    for item in value:
        if type(item) is not str:
            raise TypeError(f"{field_name} items must be str")
        if item == "":
            raise ValueError(f"{field_name} items must be non-empty")
        result.append(item)
    return result


def _build_evidence_summary(evidence: list[dict[str, Any]]) -> str:
    if not evidence:
        return "No retrieved evidence."
    return " | ".join(
        f"{item['document_id']}: {item['text']}"
        for item in evidence[:3]
    )


def _validate_json_compatible(value: object, *, field_name: str) -> None:
    _validate_json_types(value, field_name=field_name)
    try:
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
        )
    except TypeError as exc:
        raise TypeError(f"{field_name} must be JSON-compatible") from exc
    except ValueError as exc:
        raise ValueError(f"{field_name} must be JSON-compatible") from exc


def _validate_json_types(value: object, *, field_name: str) -> None:
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError(f"{field_name} must contain only finite floats")
        return
    if value_type is list:
        for item in value:
            _validate_json_types(item, field_name=field_name)
        return
    if value_type is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{field_name} dict keys must be str")
            _validate_json_types(item, field_name=field_name)
        return
    raise TypeError(f"{field_name} must be JSON-compatible")
