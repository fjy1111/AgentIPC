from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentipc.utils import utc_timestamp


NonEmptyStr = Annotated[str, Field(min_length=1, strict=True)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
StrictNonNegativeInt = Annotated[int, Field(ge=0, strict=True)]
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class MemoryType(str, Enum):
    EVIDENCE = "evidence"
    EXPERIENCE = "experience"
    RESULT = "result"


class MemoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: NonEmptyStr
    source_agent: NonEmptyStr
    created_at: NonNegativeFiniteFloat = Field(default_factory=utc_timestamp)
    task_topic: NonEmptyStr
    summary: NonEmptyStr

    memory_type: MemoryType
    tags: list[NonEmptyStr] = Field(default_factory=list)
    keywords: list[NonEmptyStr] = Field(default_factory=list)
    embedding: list[FiniteFloat] | None = None
    payload: dict[str, Any] = Field(default_factory=dict)

    reuse_count: StrictNonNegativeInt = 0
    success_count: StrictNonNegativeInt = 0
    failure_count: StrictNonNegativeInt = 0
    last_accessed_at: NonNegativeFiniteFloat | None = None

    @field_validator("tags", "keywords", mode="before")
    @classmethod
    def _require_string_list(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("value must be a list[str]")
        return value

    @field_validator("embedding", mode="before")
    @classmethod
    def _require_embedding_list(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("embedding must be a list[float] or None")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise ValueError("embedding items must be numeric")
        return value
