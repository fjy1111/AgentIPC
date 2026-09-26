import numpy as np
import pytest
from pydantic import ValidationError

from agentipc.providers.base import EmbeddingProvider, LLMProvider, LLMResponse


class FakeLLMProvider:
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        return LLMResponse(
            text="fake",
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=0.0,
            raw=None,
        )


class FakeEmbeddingProvider:
    @property
    def dim(self) -> int:
        return 4

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.zeros((len(texts), self.dim), dtype=np.float32)


def test_llm_response_constructs() -> None:
    response = LLMResponse(
        text="ok",
        prompt_tokens=3,
        completion_tokens=2,
        latency_ms=1.5,
        raw={"provider": "test"},
    )

    assert response.text == "ok"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2
    assert response.latency_ms == 1.5
    assert response.raw == {"provider": "test"}


def test_llm_response_allows_none_token_counts() -> None:
    response = LLMResponse(
        text="ok",
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=0.0,
        raw=None,
    )

    assert response.prompt_tokens is None
    assert response.completion_tokens is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("prompt_tokens", -1),
        ("completion_tokens", -1),
    ],
)
def test_llm_response_rejects_negative_token_counts(field: str, value: int) -> None:
    kwargs = {
        "text": "ok",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": 0.0,
        "raw": None,
    }
    kwargs[field] = value

    with pytest.raises(ValidationError):
        LLMResponse(**kwargs)


@pytest.mark.parametrize("latency", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_llm_response_rejects_invalid_latency(latency: float) -> None:
    with pytest.raises(ValidationError):
        LLMResponse(
            text="ok",
            prompt_tokens=0,
            completion_tokens=0,
            latency_ms=latency,
            raw=None,
        )


def test_llm_response_rejects_extra_field() -> None:
    with pytest.raises(ValidationError):
        LLMResponse(
            text="ok",
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=0.0,
            raw=None,
            provider="extra",  # type: ignore[call-arg]
        )


def test_fake_llm_provider_structurally_satisfies_protocol() -> None:
    assert isinstance(FakeLLMProvider(), LLMProvider)


def test_fake_embedding_provider_structurally_satisfies_protocol() -> None:
    provider = FakeEmbeddingProvider()

    assert isinstance(provider, EmbeddingProvider)
    embedded = provider.embed(["a", "b"])
    assert embedded.shape == (2, provider.dim)
    assert embedded.dtype == np.float32
