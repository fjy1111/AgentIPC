from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentipc.protocol.enums import (
    PROTOCOL_VERSION,
    ActionType,
    MessageStatus,
    MessageType,
)
from agentipc.protocol.refs import ArtifactRef, MemoryRef, StateRef
from agentipc.utils import new_message_id, new_task_id, new_trace_id, utc_timestamp


class AgentEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(default=PROTOCOL_VERSION, min_length=1)
    message_id: str = Field(default_factory=new_message_id, min_length=1)
    trace_id: str = Field(default_factory=new_trace_id, min_length=1)
    task_id: str = Field(default_factory=new_task_id, min_length=1)
    step_id: str = Field(min_length=1)

    sender: str = Field(min_length=1)
    receiver: str = Field(min_length=1)
    message_type: MessageType
    action: ActionType | None = None
    capability: str | None = None

    args: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None

    state_refs: list[StateRef] = Field(default_factory=list)
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    memory_refs: list[MemoryRef] = Field(default_factory=list)

    status: MessageStatus = MessageStatus.PENDING
    created_at: float = Field(default_factory=utc_timestamp)
    metrics: dict[str, Any] = Field(default_factory=dict)