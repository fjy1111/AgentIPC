from __future__ import annotations

from collections import UserDict
from dataclasses import FrozenInstanceError

import pytest

from agentipc.config import AgentIPCConfig
from agentipc.providers import factory
from agentipc.providers.base import EmbeddingProvider, LLMProvider
from agentipc.providers.factory import ProviderBundle, create_provider_bundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider


def test_default_factory_returns_working_offline_bundle() -> None:
    bundle = create_provider_bundle(AgentIPCConfig())

    assert isinstance(bundle, ProviderBundle)
    assert isinstance(bundle.llm, MockLLMProvider)
    assert isinstance(bundle.embedding, HashEmbeddingProvider)

    response = bundle.llm.complete([{"role": "user", "content": "hello"}])
    vectors = bundle.embedding.embed(["hello"])

    assert isinstance(response.text, str)
    assert vectors.shape == (1, bundle.embedding.dim)


def test_default_bundle_is_protocol_compatible() -> None:
    bundle = create_provider_bundle(AgentIPCConfig())

    assert isinstance(bundle.llm, LLMProvider)
    assert isinstance(bundle.embedding, EmbeddingProvider)


def test_provider_bundle_is_frozen() -> None:
    bundle = create_provider_bundle(AgentIPCConfig())

    with pytest.raises(FrozenInstanceError):
        bundle.llm = bundle.llm  # type: ignore[misc]


def test_default_path_does_not_construct_optional_providers(monkeypatch) -> None:
    def bomb(*args, **kwargs):
        raise AssertionError("optional provider must not be constructed")

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", bomb)
    monkeypatch.setattr(factory, "SentenceTransformerEmbeddingProvider", bomb)

    bundle = create_provider_bundle(AgentIPCConfig())

    assert isinstance(bundle.llm, MockLLMProvider)
    assert isinstance(bundle.embedding, HashEmbeddingProvider)


def test_mock_options_are_forwarded() -> None:
    bundle = create_provider_bundle(
        AgentIPCConfig(),
        llm_options={
            "default_text": "offline",
            "keyword_responses": {"plan": "planned"},
        },
    )

    matched = bundle.llm.complete([{"role": "user", "content": "make a plan"}])
    fallback = bundle.llm.complete([{"role": "user", "content": "hello"}])

    assert matched.text == "planned"
    assert fallback.text == "offline"


def test_hash_options_are_forwarded() -> None:
    bundle = create_provider_bundle(
        AgentIPCConfig(),
        embedding_options={"dim": 32},
    )

    vectors = bundle.embedding.embed(["hello"])

    assert bundle.embedding.dim == 32
    assert vectors.shape == (1, 32)


def test_openai_options_are_forwarded_without_real_sdk(monkeypatch) -> None:
    received: dict[str, object] = {}
    sentinel = object()

    def fake_openai(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", fake_openai)

    bundle = create_provider_bundle(
        AgentIPCConfig(llm_provider="openai", embedding_provider="hash"),
        llm_options={
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "http://localhost:1234/v1",
            "timeout_sec": 12,
        },
    )

    assert bundle.llm is sentinel
    assert received == {
        "model": "test-model",
        "api_key": "test-key",
        "base_url": "http://localhost:1234/v1",
        "timeout_sec": 12,
    }


def test_sentence_transformer_options_are_forwarded_without_loading_model(
    monkeypatch,
) -> None:
    received: dict[str, object] = {}
    sentinel = object()

    def fake_sentence_transformer(**kwargs):
        received.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        factory,
        "SentenceTransformerEmbeddingProvider",
        fake_sentence_transformer,
    )

    bundle = create_provider_bundle(
        AgentIPCConfig(
            llm_provider="mock",
            embedding_provider="sentence-transformer",
        ),
        embedding_options={
            "model_name": "local-test-model",
            "device": "cpu",
            "local_files_only": True,
        },
    )

    assert bundle.embedding is sentinel
    assert received == {
        "model_name": "local-test-model",
        "device": "cpu",
        "local_files_only": True,
    }


def test_both_optional_provider_paths_can_be_selected_without_network(
    monkeypatch,
) -> None:
    llm_sentinel = object()
    embedding_sentinel = object()
    llm_received: dict[str, object] = {}
    embedding_received: dict[str, object] = {}

    def fake_openai(**kwargs):
        llm_received.update(kwargs)
        return llm_sentinel

    def fake_sentence_transformer(**kwargs):
        embedding_received.update(kwargs)
        return embedding_sentinel

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", fake_openai)
    monkeypatch.setattr(
        factory,
        "SentenceTransformerEmbeddingProvider",
        fake_sentence_transformer,
    )

    bundle = create_provider_bundle(
        AgentIPCConfig(
            llm_provider="openai",
            embedding_provider="sentence-transformer",
        ),
        llm_options={"model": "test-model"},
        embedding_options={"model_name": "local-test-model"},
    )

    assert bundle.llm is llm_sentinel
    assert bundle.embedding is embedding_sentinel
    assert llm_received == {"model": "test-model"}
    assert embedding_received == {"model_name": "local-test-model"}


@pytest.mark.parametrize("options", [None, {}])
def test_openai_requires_explicit_model(options) -> None:
    with pytest.raises(
        ValueError,
        match=r"openai provider requires llm_options\['model'\]",
    ):
        create_provider_bundle(
            AgentIPCConfig(llm_provider="openai", embedding_provider="hash"),
            llm_options=options,
        )


@pytest.mark.parametrize("options", [None, {}])
def test_sentence_transformer_requires_explicit_model_name(options) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"sentence-transformer provider requires "
            r"embedding_options\['model_name'\]"
        ),
    ):
        create_provider_bundle(
            AgentIPCConfig(
                llm_provider="mock",
                embedding_provider="sentence-transformer",
            ),
            embedding_options=options,
        )


@pytest.mark.parametrize("provider", ["unknown", "Mock", " openai "])
def test_unknown_llm_provider_is_rejected(provider: str) -> None:
    with pytest.raises(ValueError, match=provider.strip() or provider):
        create_provider_bundle(AgentIPCConfig(llm_provider=provider))


@pytest.mark.parametrize(
    "provider",
    ["unknown", "Hash", " sentence-transformer ", "sentence_transformers"],
)
def test_unknown_embedding_provider_is_rejected(provider: str) -> None:
    with pytest.raises(ValueError):
        create_provider_bundle(AgentIPCConfig(embedding_provider=provider))


def test_unknown_provider_error_lists_allowed_values() -> None:
    with pytest.raises(ValueError) as exc_info:
        create_provider_bundle(AgentIPCConfig(llm_provider="unknown"))

    message = str(exc_info.value)
    assert "unknown" in message
    assert "mock" in message
    assert "openai" in message


@pytest.mark.parametrize("bad_options", [[], (), "options", 1])
def test_llm_options_must_be_mapping_or_none(bad_options) -> None:
    with pytest.raises(TypeError):
        create_provider_bundle(AgentIPCConfig(), llm_options=bad_options)


@pytest.mark.parametrize("bad_options", [[], (), "options", 1])
def test_embedding_options_must_be_mapping_or_none(bad_options) -> None:
    with pytest.raises(TypeError):
        create_provider_bundle(AgentIPCConfig(), embedding_options=bad_options)


def test_non_string_llm_option_key_is_rejected() -> None:
    with pytest.raises(TypeError):
        create_provider_bundle(AgentIPCConfig(), llm_options={1: "bad"})


def test_non_string_embedding_option_key_is_rejected() -> None:
    with pytest.raises(TypeError):
        create_provider_bundle(AgentIPCConfig(), embedding_options={1: "bad"})


@pytest.mark.parametrize(
    ("config", "llm_options", "embedding_options"),
    [
        (AgentIPCConfig(), {"model": "gpt-x"}, None),
        (AgentIPCConfig(), None, {"model_name": "model"}),
        (
            AgentIPCConfig(llm_provider="openai", embedding_provider="hash"),
            {"model": "gpt-x", "unknown": 1},
            None,
        ),
        (
            AgentIPCConfig(
                llm_provider="mock",
                embedding_provider="sentence-transformer",
            ),
            None,
            {"model_name": "model", "unknown": 1},
        ),
    ],
)
def test_backend_specific_unknown_options_are_rejected(
    config: AgentIPCConfig,
    llm_options,
    embedding_options,
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        create_provider_bundle(
            config,
            llm_options=llm_options,
            embedding_options=embedding_options,
        )


def test_provider_validation_exception_propagates() -> None:
    with pytest.raises(ValueError):
        create_provider_bundle(
            AgentIPCConfig(),
            embedding_options={"dim": 0},
        )


def test_config_must_be_agentipc_config() -> None:
    with pytest.raises(TypeError):
        create_provider_bundle({})  # type: ignore[arg-type]


def test_options_are_copied_before_construction() -> None:
    llm_options = UserDict({"default_text": "before"})
    embedding_options = UserDict({"dim": 32})

    bundle = create_provider_bundle(
        AgentIPCConfig(),
        llm_options=llm_options,
        embedding_options=embedding_options,
    )

    llm_options["default_text"] = "after"
    embedding_options["dim"] = 64

    response = bundle.llm.complete([{"role": "user", "content": "hello"}])
    assert response.text == "before"
    assert bundle.embedding.dim == 32
