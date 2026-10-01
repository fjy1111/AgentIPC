from __future__ import annotations

import math
import time
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator


StrictNonNegativeInt = Annotated[int, Field(ge=0, strict=True)]


class MetricsSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message_count: StrictNonNegativeInt = 0
    text_chars: StrictNonNegativeInt = 0
    text_tokens: StrictNonNegativeInt = 0
    wire_chars: StrictNonNegativeInt = 0
    wire_tokens: StrictNonNegativeInt = 0
    wire_bytes: StrictNonNegativeInt = 0
    protocol_bytes: StrictNonNegativeInt = 0

    state_transfer_count: StrictNonNegativeInt = 0
    state_bytes: StrictNonNegativeInt = 0

    artifact_ref_count: StrictNonNegativeInt = 0
    artifact_payload_bytes: StrictNonNegativeInt = 0

    memory_retrieved: StrictNonNegativeInt = 0
    memory_used: StrictNonNegativeInt = 0
    memory_effective: StrictNonNegativeInt = 0
    memory_harmful: StrictNonNegativeInt = 0
    fast_path_hit_count: StrictNonNegativeInt = 0

    tool_call_count: StrictNonNegativeInt = 0
    repeated_tool_call_count: StrictNonNegativeInt = 0

    llm_call_count: StrictNonNegativeInt = 0
    llm_prompt_tokens: StrictNonNegativeInt = 0
    llm_completion_tokens: StrictNonNegativeInt = 0
    llm_total_tokens: StrictNonNegativeInt = 0
    llm_usage_missing_count: StrictNonNegativeInt = 0
    llm_latency_ms: float = 0.0

    latency_ms: float = 0.0
    success: StrictBool = False

    @field_validator("latency_ms", mode="before")
    @classmethod
    def _validate_latency_ms(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("latency_ms must be an int or float")

        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("latency_ms must be finite")
        if normalized < 0:
            raise ValueError("latency_ms must be non-negative")
        return normalized

    @field_validator("llm_latency_ms", mode="before")
    @classmethod
    def _validate_llm_latency_ms(cls, value: object) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("llm_latency_ms must be an int or float")

        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("llm_latency_ms must be finite")
        if normalized < 0:
            raise ValueError("llm_latency_ms must be non-negative")
        return normalized


class MetricsCollector:
    _INCREMENTABLE_METRICS: ClassVar[frozenset[str]] = frozenset(
        {
            "message_count",
            "text_chars",
            "text_tokens",
            "wire_chars",
            "wire_tokens",
            "wire_bytes",
            "protocol_bytes",
            "state_transfer_count",
            "state_bytes",
            "artifact_ref_count",
            "artifact_payload_bytes",
            "memory_retrieved",
            "memory_used",
            "memory_effective",
            "memory_harmful",
            "fast_path_hit_count",
            "tool_call_count",
            "repeated_tool_call_count",
        }
    )

    def __init__(self) -> None:
        self._counters = {metric: 0 for metric in self._INCREMENTABLE_METRICS}
        self._latency_ms = 0.0
        self._success = False
        self._llm_call_count = 0
        self._llm_prompt_tokens = 0
        self._llm_completion_tokens = 0
        self._llm_total_tokens = 0
        self._llm_usage_missing_count = 0
        self._llm_latency_ms = 0.0

    def increment(self, metric: str, amount: int = 1) -> None:
        if not isinstance(metric, str):
            raise TypeError("metric must be a string")
        if metric not in self._INCREMENTABLE_METRICS:
            raise ValueError(f"metric is not incrementable: {metric}")
        if type(amount) is not int:
            raise TypeError("amount must be an int")
        if amount < 0:
            raise ValueError("amount must be non-negative")

        self._counters[metric] += amount

    def set_latency_ms(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("latency_ms must be an int or float")

        normalized = float(value)
        if not math.isfinite(normalized):
            raise ValueError("latency_ms must be finite")
        if normalized < 0:
            raise ValueError("latency_ms must be non-negative")

        self._latency_ms = normalized

    def set_success(self, success: bool) -> None:
        if type(success) is not bool:
            raise TypeError("success must be a bool")
        self._success = success

    def record_llm_response(self, response: object) -> None:
        from agentipc.providers.base import LLMResponse

        if not isinstance(response, LLMResponse):
            raise TypeError("response must be an LLMResponse")

        self._llm_call_count += 1
        self._llm_latency_ms += response.latency_ms

        if (
            response.prompt_tokens is not None
            and response.completion_tokens is not None
        ):
            self._llm_prompt_tokens += response.prompt_tokens
            self._llm_completion_tokens += response.completion_tokens
            self._llm_total_tokens += (
                response.prompt_tokens + response.completion_tokens
            )
        else:
            self._llm_usage_missing_count += 1

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            **self._counters,
            llm_call_count=self._llm_call_count,
            llm_prompt_tokens=self._llm_prompt_tokens,
            llm_completion_tokens=self._llm_completion_tokens,
            llm_total_tokens=self._llm_total_tokens,
            llm_usage_missing_count=self._llm_usage_missing_count,
            llm_latency_ms=self._llm_latency_ms,
            latency_ms=self._latency_ms,
            success=self._success,
        )


class TaskTimer:
    def __init__(self) -> None:
        self._started_ns: int | None = None
        self._completed = False
        self._latency_ms = 0.0

    def start(self) -> None:
        if self._completed:
            raise RuntimeError("timer has already completed and cannot be restarted")
        if self._started_ns is not None:
            return

        self._started_ns = time.perf_counter_ns()

    def stop(self) -> float:
        if self._started_ns is None:
            raise RuntimeError("timer has not been started")
        if self._completed:
            return self._latency_ms

        stopped_ns = time.perf_counter_ns()
        if stopped_ns < self._started_ns:
            raise RuntimeError("timer clock moved backwards")

        self._latency_ms = (stopped_ns - self._started_ns) / 1_000_000.0
        self._completed = True
        return self._latency_ms

    @property
    def latency_ms(self) -> float:
        return self._latency_ms
