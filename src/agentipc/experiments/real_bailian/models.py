from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class NormalizedKnowledgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: StrictBool
    matched_answer_contains: list[str]
    missing_answer_contains: list[str]


class MemoryValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_used: int = Field(ge=0)
    runtime_memory_effective_strict: int = Field(ge=0)
    runtime_memory_harmful_strict: int = Field(ge=0)
    validated_memory_effective: int = Field(ge=0)
    validated_memory_harmful: int = Field(ge=0)
    strict_vs_validated_discrepancy: StrictBool
