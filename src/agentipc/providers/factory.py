from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agentipc.config import AgentIPCConfig
from agentipc.providers.base import EmbeddingProvider, LLMProvider
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.providers.openai_compatible import OpenAICompatibleProvider
from agentipc.providers.openai_embedding import OpenAICompatibleEmbeddingProvider
from agentipc.providers.sentence_transformer import SentenceTransformerEmbeddingProvider

_LLM_PROVIDERS = ("mock", "openai")
_EMBEDDING_PROVIDERS = ("hash", "sentence-transformer", "openai")

_MOCK_OPTIONS = frozenset({"keyword_responses", "default_text"})
_HASH_OPTIONS = frozenset({"dim"})
_OPENAI_OPTIONS = frozenset({"model", "api_key", "base_url", "timeout_sec"})
_OPENAI_EMBEDDING_OPTIONS = frozenset({"model", "dim", "api_key", "base_url", "timeout_sec"})
_SENTENCE_TRANSFORMER_OPTIONS = frozenset(
    {"model_name", "device", "local_files_only"}
)


@dataclass(frozen=True)
class ProviderBundle:
    llm: LLMProvider
    embedding: EmbeddingProvider


def create_provider_bundle(
    config: AgentIPCConfig,
    *,
    llm_options: Mapping[str, object] | None = None,
    embedding_options: Mapping[str, object] | None = None,
) -> ProviderBundle:
    if not isinstance(config, AgentIPCConfig):
        raise TypeError("config must be an AgentIPCConfig instance")

    llm_kwargs = _copy_options(llm_options, name="llm_options")
    embedding_kwargs = _copy_options(
        embedding_options,
        name="embedding_options",
    )

    llm = _create_llm(config.llm_provider, llm_kwargs)
    embedding = _create_embedding(config.embedding_provider, embedding_kwargs)

    return ProviderBundle(llm=llm, embedding=embedding)


def _copy_options(
    options: Mapping[str, object] | None,
    *,
    name: str,
) -> dict[str, object]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"{name} must be a Mapping[str, object] or None")

    copied: dict[str, object] = {}
    for key, value in options.items():
        if not isinstance(key, str):
            raise TypeError(f"all {name} keys must be str")
        copied[key] = value
    return copied


def _reject_unknown_options(
    options: Mapping[str, object],
    *,
    provider: str,
    option_group: str,
    allowed: frozenset[str],
) -> None:
    unknown = sorted(set(options) - allowed)
    if unknown:
        allowed_text = ", ".join(sorted(allowed))
        unknown_text = ", ".join(repr(key) for key in unknown)
        raise ValueError(
            f"unknown {option_group} for {provider} provider: {unknown_text}; "
            f"allowed options: {allowed_text}"
        )


def _create_llm(provider: str, options: dict[str, object]) -> LLMProvider:
    if provider == "mock":
        _reject_unknown_options(
            options,
            provider=provider,
            option_group="llm_options",
            allowed=_MOCK_OPTIONS,
        )
        return MockLLMProvider(**options)

    if provider == "openai":
        _reject_unknown_options(
            options,
            provider=provider,
            option_group="llm_options",
            allowed=_OPENAI_OPTIONS,
        )
        if "model" not in options:
            raise ValueError("openai provider requires llm_options['model']")
        return OpenAICompatibleProvider(**options)

    allowed_text = ", ".join(_LLM_PROVIDERS)
    raise ValueError(
        f"unknown LLM provider {provider!r}; allowed values: {allowed_text}"
    )


def _create_embedding(
    provider: str,
    options: dict[str, object],
) -> EmbeddingProvider:
    if provider == "hash":
        _reject_unknown_options(
            options,
            provider=provider,
            option_group="embedding_options",
            allowed=_HASH_OPTIONS,
        )
        return HashEmbeddingProvider(**options)

    if provider == "sentence-transformer":
        _reject_unknown_options(
            options,
            provider=provider,
            option_group="embedding_options",
            allowed=_SENTENCE_TRANSFORMER_OPTIONS,
        )
        if "model_name" not in options:
            raise ValueError(
                "sentence-transformer provider requires "
                "embedding_options['model_name']"
            )
        return SentenceTransformerEmbeddingProvider(**options)

    if provider == "openai":
        _reject_unknown_options(
            options,
            provider=provider,
            option_group="embedding_options",
            allowed=_OPENAI_EMBEDDING_OPTIONS,
        )
        if "model" not in options:
            raise ValueError(
                "openai embedding provider requires embedding_options['model']"
            )
        if "dim" not in options:
            raise ValueError(
                "openai embedding provider requires embedding_options['dim']"
            )
        return OpenAICompatibleEmbeddingProvider(**options)

    allowed_text = ", ".join(_EMBEDDING_PROVIDERS)
    raise ValueError(
        f"unknown embedding provider {provider!r}; allowed values: {allowed_text}"
    )
