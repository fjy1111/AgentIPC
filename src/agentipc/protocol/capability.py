from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentipc.protocol.enums import PROTOCOL_VERSION


class AgentCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(min_length=1)
    capabilities: list[str] = Field(default_factory=list)
    protocol_versions: list[str] = Field(
        default_factory=lambda: [PROTOCOL_VERSION]
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("capabilities", "protocol_versions")
    @classmethod
    def reject_empty_entries(cls, values: list[str]) -> list[str]:
        if any(value == "" for value in values):
            raise ValueError("entries must be non-empty strings")
        return values