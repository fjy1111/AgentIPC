from __future__ import annotations

import time
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from agentipc.providers.base import EmbeddingProvider, LLMProvider, LLMResponse


class LLMCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_index: int = Field(ge=1)
    provider: str
    model: str
    request_id: str | None = None
    finish_reason: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0.0)


class EmbeddingCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_index: int = Field(ge=1)
    provider: str
    model: str
    dim: int = Field(ge=1)
    input_count: int = Field(ge=0)
    input_chars: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)


class ProviderUsage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_calls: list[LLMCallRecord]
    embedding_calls: list[EmbeddingCallRecord]
    llm_call_count: int = Field(ge=0)
    llm_prompt_tokens: int = Field(ge=0)
    llm_completion_tokens: int = Field(ge=0)
    llm_total_tokens: int = Field(ge=0)
    llm_usage_missing_count: int = Field(ge=0)
    llm_latency_ms: float = Field(ge=0.0)
    embedding_call_count: int = Field(ge=0)
    embedding_input_count: int = Field(ge=0)
    embedding_input_chars: int = Field(ge=0)
    embedding_latency_ms: float = Field(ge=0.0)


class RecordingLLMProvider:
    """Non-mutating experiment decorator for an existing LLM provider."""

    def __init__(
        self,
        delegate: LLMProvider,
        *,
        model: str,
        provider_name: str = "openai_compatible",
    ) -> None:
        if not isinstance(delegate, LLMProvider):
            raise TypeError("delegate must satisfy LLMProvider")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty str")
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError("provider_name must be a non-empty str")
        self._delegate = delegate
        self._model = model
        self._provider_name = provider_name
        self._records: list[LLMCallRecord] = []

    @property
    def records(self) -> tuple[LLMCallRecord, ...]:
        return tuple(self._records)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        response = self._delegate.complete(messages, temperature=temperature)
        if not isinstance(response, LLMResponse):
            raise TypeError("delegate.complete() must return LLMResponse")

        raw = response.raw if isinstance(response.raw, dict) else {}
        provider = _optional_str(raw.get("provider")) or self._provider_name
        model = _optional_str(raw.get("model")) or self._model
        request_id = _optional_str(raw.get("request_id"))
        finish_reason = _optional_str(raw.get("finish_reason"))

        if response.prompt_tokens is not None and response.completion_tokens is not None:
            total_tokens = response.prompt_tokens + response.completion_tokens
        else:
            total_tokens = None

        self._records.append(
            LLMCallRecord(
                call_index=len(self._records) + 1,
                provider=provider,
                model=model,
                request_id=request_id,
                finish_reason=finish_reason,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=total_tokens,
                latency_ms=response.latency_ms,
            )
        )
        return response


class RecordingEmbeddingProvider:
    """Experiment decorator that records shape-neutral embedding usage."""

    def __init__(
        self,
        delegate: EmbeddingProvider,
        *,
        model: str,
        provider_name: str = "openai_compatible",
    ) -> None:
        if not isinstance(delegate, EmbeddingProvider):
            raise TypeError("delegate must satisfy EmbeddingProvider")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be a non-empty str")
        if not isinstance(provider_name, str) or not provider_name:
            raise ValueError("provider_name must be a non-empty str")
        self._delegate = delegate
        self._model = model
        self._provider_name = provider_name
        self._records: list[EmbeddingCallRecord] = []

    @property
    def dim(self) -> int:
        return self._delegate.dim

    @property
    def records(self) -> tuple[EmbeddingCallRecord, ...]:
        return tuple(self._records)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list[str]")
        started = time.perf_counter()
        matrix = self._delegate.embed(texts)
        latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)

        self._records.append(
            EmbeddingCallRecord(
                call_index=len(self._records) + 1,
                provider=self._provider_name,
                model=self._model,
                dim=self.dim,
                input_count=len(texts),
                input_chars=sum(len(text) for text in texts),
                latency_ms=latency_ms,
            )
        )
        return matrix


def build_provider_usage(
    llm: RecordingLLMProvider,
    embedding: RecordingEmbeddingProvider,
) -> ProviderUsage:
    llm_records = list(llm.records)
    embedding_records = list(embedding.records)

    prompt_tokens = sum(record.prompt_tokens or 0 for record in llm_records)
    completion_tokens = sum(record.completion_tokens or 0 for record in llm_records)
    total_tokens = sum(record.total_tokens or 0 for record in llm_records)
    usage_missing = sum(
        1
        for record in llm_records
        if record.prompt_tokens is None or record.completion_tokens is None
    )

    return ProviderUsage(
        llm_calls=llm_records,
        embedding_calls=embedding_records,
        llm_call_count=len(llm_records),
        llm_prompt_tokens=prompt_tokens,
        llm_completion_tokens=completion_tokens,
        llm_total_tokens=total_tokens,
        llm_usage_missing_count=usage_missing,
        llm_latency_ms=sum(record.latency_ms for record in llm_records),
        embedding_call_count=len(embedding_records),
        embedding_input_count=sum(record.input_count for record in embedding_records),
        embedding_input_chars=sum(record.input_chars for record in embedding_records),
        embedding_latency_ms=sum(record.latency_ms for record in embedding_records),
    )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) else str(value)
