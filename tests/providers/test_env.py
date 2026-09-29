"""Tests for environment-driven provider configuration."""

from __future__ import annotations

import pytest

from agentipc.config import AgentIPCConfig
from agentipc.providers import factory
from agentipc.providers.env import ResolvedProviderBundle, create_provider_bundle_from_env
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider


def test_default_offline_with_empty_environ() -> None:
    """Default config with no environment should give offline mock/hash bundle."""
    resolved = create_provider_bundle_from_env(environ={})

    assert isinstance(resolved, ResolvedProviderBundle)
    assert resolved.config.llm_provider == "mock"
    assert resolved.config.embedding_provider == "hash"
    assert isinstance(resolved.bundle.llm, MockLLMProvider)
    assert isinstance(resolved.bundle.embedding, HashEmbeddingProvider)


def test_default_config_when_none_provided() -> None:
    """create_provider_bundle_from_env with config=None should use defaults."""
    resolved = create_provider_bundle_from_env(config=None, environ={})

    assert resolved.config.llm_provider == "mock"
    assert resolved.config.embedding_provider == "hash"
    assert isinstance(resolved.bundle.llm, MockLLMProvider)


def test_original_config_unchanged_after_env_override() -> None:
    """Environment overrides should not mutate the original config."""
    original = AgentIPCConfig()
    assert original.llm_provider == "mock"

    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "test-model",
    }

    # Mock OpenAI provider construction
    sentinel = object()

    def fake_openai(**kwargs):
        return sentinel

    import agentipc.providers.env as env_module

    original_factory_module = env_module.create_provider_bundle.__globals__["factory"]
    original_openai_class = original_factory_module.OpenAICompatibleProvider

    try:
        original_factory_module.OpenAICompatibleProvider = fake_openai
        resolved = create_provider_bundle_from_env(config=original, environ=environ)

        # Resolved config reflects env override
        assert resolved.config.llm_provider == "openai"

        # Original config unchanged
        assert original.llm_provider == "mock"
    finally:
        original_factory_module.OpenAICompatibleProvider = original_openai_class


def test_cloud_openai_endpoint(monkeypatch) -> None:
    """Cloud OpenAI-compatible provider with full configuration."""
    received_kwargs: dict[str, object] = {}
    sentinel = object()

    def fake_openai(**kwargs):
        received_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", fake_openai)

    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "cloud-model",
        "AGENTIPC_LLM_API_KEY": "cloud-secret",
        "AGENTIPC_LLM_BASE_URL": "https://api.example.test/v1",
        "AGENTIPC_LLM_TIMEOUT_SEC": "12.5",
    }

    resolved = create_provider_bundle_from_env(environ=environ)

    assert resolved.bundle.llm is sentinel
    assert received_kwargs == {
        "model": "cloud-model",
        "api_key": "cloud-secret",
        "base_url": "https://api.example.test/v1",
        "timeout_sec": 12.5,
    }


def test_local_openai_compatible_endpoint(monkeypatch) -> None:
    """Local OpenAI-compatible server uses same provider as cloud."""
    received_kwargs: dict[str, object] = {}
    sentinel = object()

    def fake_openai(**kwargs):
        received_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", fake_openai)

    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "local-model",
        "AGENTIPC_LLM_API_KEY": "local",
        "AGENTIPC_LLM_BASE_URL": "http://127.0.0.1:8000/v1",
    }

    resolved = create_provider_bundle_from_env(environ=environ)

    assert resolved.bundle.llm is sentinel
    assert received_kwargs == {
        "model": "local-model",
        "api_key": "local",
        "base_url": "http://127.0.0.1:8000/v1",
    }


def test_local_openai_without_api_key(monkeypatch) -> None:
    """Local server may not require API key."""
    received_kwargs: dict[str, object] = {}
    sentinel = object()

    def fake_openai(**kwargs):
        received_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", fake_openai)

    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "local-model",
        "AGENTIPC_LLM_BASE_URL": "http://127.0.0.1:8000/v1",
    }

    resolved = create_provider_bundle_from_env(environ=environ)

    assert resolved.bundle.llm is sentinel
    assert received_kwargs == {
        "model": "local-model",
        "base_url": "http://127.0.0.1:8000/v1",
    }
    assert "api_key" not in received_kwargs


def test_api_key_not_in_config_dump(monkeypatch) -> None:
    """API keys must not appear in AgentIPCConfig serialization."""
    sentinel = object()

    def fake_openai(**kwargs):
        return sentinel

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", fake_openai)

    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "test-model",
        "AGENTIPC_LLM_API_KEY": "super-secret-value",
    }

    resolved = create_provider_bundle_from_env(environ=environ)

    config_dict = resolved.config.model_dump()
    config_str = str(config_dict)

    assert "super-secret-value" not in config_str
    assert "api_key" not in config_dict


def test_irrelevant_openai_env_ignored_in_mock_mode() -> None:
    """Mock mode ignores OpenAI environment variables."""
    environ = {
        "AGENTIPC_LLM_MODEL": "should-be-ignored",
        "AGENTIPC_LLM_API_KEY": "should-be-ignored",
        "AGENTIPC_LLM_BASE_URL": "http://localhost:8000/v1",
    }

    resolved = create_provider_bundle_from_env(environ=environ)

    assert resolved.config.llm_provider == "mock"
    assert isinstance(resolved.bundle.llm, MockLLMProvider)


def test_llm_provider_env_override() -> None:
    """AGENTIPC_LLM_PROVIDER overrides config.llm_provider."""
    config = AgentIPCConfig(llm_provider="mock")

    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "test-model",
    }

    sentinel = object()

    def fake_openai(**kwargs):
        return sentinel

    import agentipc.providers.env as env_module

    original_factory_module = env_module.create_provider_bundle.__globals__["factory"]
    original_openai_class = original_factory_module.OpenAICompatibleProvider

    try:
        original_factory_module.OpenAICompatibleProvider = fake_openai
        resolved = create_provider_bundle_from_env(config=config, environ=environ)

        assert resolved.config.llm_provider == "openai"
    finally:
        original_factory_module.OpenAICompatibleProvider = original_openai_class


def test_embedding_provider_env_override() -> None:
    """AGENTIPC_EMBEDDING_PROVIDER overrides config.embedding_provider."""
    config = AgentIPCConfig(embedding_provider="hash")

    environ = {
        "AGENTIPC_EMBEDDING_PROVIDER": "sentence-transformer",
        "AGENTIPC_EMBEDDING_MODEL_NAME": "test-model",
    }

    sentinel = object()

    def fake_sentence_transformer(**kwargs):
        return sentinel

    import agentipc.providers.env as env_module

    original_factory_module = env_module.create_provider_bundle.__globals__["factory"]
    original_st_class = original_factory_module.SentenceTransformerEmbeddingProvider

    try:
        original_factory_module.SentenceTransformerEmbeddingProvider = (
            fake_sentence_transformer
        )
        resolved = create_provider_bundle_from_env(config=config, environ=environ)

        assert resolved.config.embedding_provider == "sentence-transformer"
    finally:
        original_factory_module.SentenceTransformerEmbeddingProvider = original_st_class


def test_hash_embedding_dim_option(monkeypatch) -> None:
    """AGENTIPC_EMBEDDING_DIM configures hash embedding dimension."""
    received_kwargs: dict[str, object] = {}
    sentinel = object()

    def fake_hash(**kwargs):
        received_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(factory, "HashEmbeddingProvider", fake_hash)

    environ = {
        "AGENTIPC_EMBEDDING_PROVIDER": "hash",
        "AGENTIPC_EMBEDDING_DIM": "96",
    }

    resolved = create_provider_bundle_from_env(environ=environ)

    assert resolved.bundle.embedding is sentinel
    assert received_kwargs == {"dim": 96}


def test_sentence_transformer_full_config(monkeypatch) -> None:
    """All sentence-transformer options from environment."""
    received_kwargs: dict[str, object] = {}
    sentinel = object()

    def fake_sentence_transformer(**kwargs):
        received_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        factory,
        "SentenceTransformerEmbeddingProvider",
        fake_sentence_transformer,
    )

    environ = {
        "AGENTIPC_EMBEDDING_PROVIDER": "sentence-transformer",
        "AGENTIPC_EMBEDDING_MODEL_NAME": "local-embedding-model",
        "AGENTIPC_EMBEDDING_DEVICE": "cpu",
        "AGENTIPC_EMBEDDING_LOCAL_FILES_ONLY": "true",
    }

    resolved = create_provider_bundle_from_env(environ=environ)

    assert resolved.bundle.embedding is sentinel
    assert received_kwargs == {
        "model_name": "local-embedding-model",
        "device": "cpu",
        "local_files_only": True,
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        ("true", True),
        ("TRUE", True),
        ("True", True),
        ("false", False),
        ("FALSE", False),
        ("False", False),
    ],
)
def test_bool_parsing_accepts_case_variants(value: str, expected: bool, monkeypatch) -> None:
    """Boolean parser accepts true/false case-insensitively."""
    received_kwargs: dict[str, object] = {}

    def fake_sentence_transformer(**kwargs):
        received_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(
        factory,
        "SentenceTransformerEmbeddingProvider",
        fake_sentence_transformer,
    )

    environ = {
        "AGENTIPC_EMBEDDING_PROVIDER": "sentence-transformer",
        "AGENTIPC_EMBEDDING_MODEL_NAME": "test-model",
        "AGENTIPC_EMBEDDING_LOCAL_FILES_ONLY": value,
    }

    create_provider_bundle_from_env(environ=environ)

    assert received_kwargs["local_files_only"] == expected


@pytest.mark.parametrize(
    "bad_value",
    ["1", "0", "yes", "no", "on", "off", "", "t", "f", "TRUE "],
)
def test_bool_parsing_rejects_non_true_false(bad_value: str, monkeypatch) -> None:
    """Boolean parser rejects values other than true/false."""
    monkeypatch.setattr(
        factory,
        "SentenceTransformerEmbeddingProvider",
        lambda **kwargs: object(),
    )

    environ = {
        "AGENTIPC_EMBEDDING_PROVIDER": "sentence-transformer",
        "AGENTIPC_EMBEDDING_MODEL_NAME": "test-model",
        "AGENTIPC_EMBEDDING_LOCAL_FILES_ONLY": bad_value,
    }

    with pytest.raises(ValueError, match="must be 'true' or 'false'"):
        create_provider_bundle_from_env(environ=environ)


def test_missing_openai_model_fails_at_factory(monkeypatch) -> None:
    """Missing model for openai provider fails in existing factory."""
    monkeypatch.setattr(
        factory,
        "OpenAICompatibleProvider",
        lambda **kwargs: object(),
    )

    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
    }

    with pytest.raises(ValueError, match="openai provider requires llm_options"):
        create_provider_bundle_from_env(environ=environ)


def test_empty_llm_provider_env_fails() -> None:
    """Empty AGENTIPC_LLM_PROVIDER must fail."""
    environ = {"AGENTIPC_LLM_PROVIDER": ""}

    with pytest.raises(ValueError, match="AGENTIPC_LLM_PROVIDER must not be empty"):
        create_provider_bundle_from_env(environ=environ)


def test_empty_embedding_provider_env_fails() -> None:
    """Empty AGENTIPC_EMBEDDING_PROVIDER must fail."""
    environ = {"AGENTIPC_EMBEDDING_PROVIDER": ""}

    with pytest.raises(ValueError, match="AGENTIPC_EMBEDDING_PROVIDER must not be empty"):
        create_provider_bundle_from_env(environ=environ)


def test_empty_llm_model_fails() -> None:
    """Empty AGENTIPC_LLM_MODEL must fail."""
    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "",
    }

    with pytest.raises(ValueError, match="AGENTIPC_LLM_MODEL must not be empty"):
        create_provider_bundle_from_env(environ=environ)


def test_empty_llm_api_key_fails() -> None:
    """Empty AGENTIPC_LLM_API_KEY must fail."""
    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "test-model",
        "AGENTIPC_LLM_API_KEY": "",
    }

    with pytest.raises(ValueError, match="AGENTIPC_LLM_API_KEY must not be empty"):
        create_provider_bundle_from_env(environ=environ)


def test_bad_timeout_value_fails() -> None:
    """Invalid timeout values must fail."""
    environ = {
        "AGENTIPC_LLM_PROVIDER": "openai",
        "AGENTIPC_LLM_MODEL": "test-model",
        "AGENTIPC_LLM_TIMEOUT_SEC": "not-a-number",
    }

    with pytest.raises(ValueError, match="AGENTIPC_LLM_TIMEOUT_SEC must be a valid number"):
        create_provider_bundle_from_env(environ=environ)


def test_bad_embedding_dim_fails() -> None:
    """Invalid dimension values must fail."""
    environ = {
        "AGENTIPC_EMBEDDING_PROVIDER": "hash",
        "AGENTIPC_EMBEDDING_DIM": "not-an-int",
    }

    with pytest.raises(ValueError, match="AGENTIPC_EMBEDDING_DIM must be a valid integer"):
        create_provider_bundle_from_env(environ=environ)


def test_config_must_be_agentipc_config() -> None:
    """config parameter must be AgentIPCConfig or None."""
    with pytest.raises(TypeError, match="config must be an AgentIPCConfig"):
        create_provider_bundle_from_env(config={}, environ={})  # type: ignore[arg-type]


def test_environ_must_be_mapping() -> None:
    """environ parameter must be Mapping or None."""
    with pytest.raises(TypeError, match="environ must be a Mapping"):
        create_provider_bundle_from_env(environ=[])  # type: ignore[arg-type]


def test_environ_values_must_be_strings() -> None:
    """All environ values must be strings."""
    with pytest.raises(TypeError, match="all environ values must be str"):
        create_provider_bundle_from_env(
            environ={"AGENTIPC_LLM_PROVIDER": 123}  # type: ignore[dict-item]
        )


def test_resolved_bundle_is_frozen() -> None:
    """ResolvedProviderBundle should be immutable."""
    from dataclasses import FrozenInstanceError

    resolved = create_provider_bundle_from_env(environ={})

    with pytest.raises(FrozenInstanceError):
        resolved.config = resolved.config  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        resolved.bundle = resolved.bundle  # type: ignore[misc]


def test_factory_reused_for_provider_construction(monkeypatch) -> None:
    """env.py must delegate to existing factory, not reimplement."""
    factory_called = False

    original_create = factory.create_provider_bundle

    def instrumented_create(*args, **kwargs):
        nonlocal factory_called
        factory_called = True
        return original_create(*args, **kwargs)

    monkeypatch.setattr(factory, "create_provider_bundle", instrumented_create)

    create_provider_bundle_from_env(environ={})

    assert factory_called, "env.py must call factory.create_provider_bundle"
