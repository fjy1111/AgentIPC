from typing import Annotated, Any, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


NonNegativeInt = Annotated[int, Field(ge=0)]
NonNegativeFiniteFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class LLMResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    prompt_tokens: NonNegativeInt | None
    completion_tokens: NonNegativeInt | None
    latency_ms: NonNegativeFiniteFloat
    raw: dict[str, Any] | None


@runtime_checkable
class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dim(self) -> int:
        ...

    def embed(
        self,
        texts: list[str],
    ) -> np.ndarray:
        ...
