from __future__ import annotations

import json
import math
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef
from agentipc.utils import utc_timestamp


class TraceEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_type: str = Field(min_length=1)
    recorded_at: float

    trace_id: str = Field(min_length=1)
    task_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)
    step_id: str = Field(min_length=1)

    sender: str = Field(min_length=1)
    receiver: str = Field(min_length=1)

    message_type: str = Field(min_length=1)
    action: str | None
    status: str = Field(min_length=1)

    state_refs: list[StateRef]
    artifact_refs: list[ArtifactRef]
    memory_refs: list[MemoryRef]

    @field_validator("recorded_at", mode="before")
    @classmethod
    def _validate_recorded_at(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("recorded_at must be an int or float")
        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("recorded_at must be finite")
        if normalized < 0:
            raise ValueError("recorded_at must be non-negative")
        return normalized


class TraceLogger:
    def __init__(self, path: str | Path) -> None:
        if not isinstance(path, (str, Path)):
            raise TypeError("path must be a string or Path")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def log_envelope(self, envelope: AgentEnvelope) -> TraceEvent:
        if not isinstance(envelope, AgentEnvelope):
            raise TypeError("envelope must be an AgentEnvelope")

        event = TraceEvent(
            event_type="message",
            recorded_at=utc_timestamp(),
            trace_id=envelope.trace_id,
            task_id=envelope.task_id,
            message_id=envelope.message_id,
            step_id=envelope.step_id,
            sender=envelope.sender,
            receiver=envelope.receiver,
            message_type=envelope.message_type.value,
            action=None if envelope.action is None else envelope.action.value,
            status=envelope.status.value,
            state_refs=list(envelope.state_refs),
            artifact_refs=list(envelope.artifact_refs),
            memory_refs=list(envelope.memory_refs),
        )
        line = json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
            stream.write("\n")
        return event
