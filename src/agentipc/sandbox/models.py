from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SandboxResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_code: int = Field(strict=True)
    stdout: str = Field(strict=True)
    stderr: str = Field(strict=True)
    timed_out: bool = Field(strict=True)
    duration_ms: float

    @field_validator("duration_ms", mode="before")
    @classmethod
    def _validate_duration_ms(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("duration_ms must be an int or float")

        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("duration_ms must be finite")
        if normalized < 0:
            raise ValueError("duration_ms must be non-negative")
        return normalized
