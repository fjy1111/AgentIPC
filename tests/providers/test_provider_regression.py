from __future__ import annotations

import numpy as np

from agentipc.config import AgentIPCConfig
from agentipc.providers import factory
from agentipc.providers.base import EmbeddingProvider, LLMProvider
from agentipc.providers.factory import ProviderBundle, create_provider_bundle
from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider
from agentipc.providers.openai_compatible import OpenAICompatibleProvider
from agentipc.providers.sentence_transformer import SentenceTransformerEmbeddingProvider


def test_provider_modules_are_import_safe() -> None:
    assert LLMProvider is not None
    assert EmbeddingProvider is not None
    assert MockLLMProvider is not None
    assert HashEmbeddingProvider is not None
    assert OpenAICompatibleProvider is not None
    assert SentenceTransformerEmbeddingProvider is not None
    assert ProviderBundle is not None
    assert create_provider_bundle is not None


def test_default_config_provider_regression_is_fully_offline() -> None:
    config = AgentIPCConfig()
    bundle = create_provider_bundle(config)

    response = bundle.llm.complete(
        [{"role": "user", "content": "offline smoke"}]
    )
    vectors = bundle.embedding.embed(
        ["linux shared memory", "artifact store"]
    )

    assert isinstance(response.text, str)
    assert vectors.shape == (2, bundle.embedding.dim)
    assert vectors.dtype == np.float32
    assert np.isfinite(vectors).all()


def test_default_smoke_does_not_touch_optional_provider_constructors(
    monkeypatch,
) -> None:
    def bomb(*args, **kwargs):
        raise AssertionError("optional provider constructor was touched")

    monkeypatch.setattr(factory, "OpenAICompatibleProvider", bomb)
    monkeypatch.setattr(factory, "SentenceTransformerEmbeddingProvider", bomb)

    bundle = create_provider_bundle(AgentIPCConfig())
    response = bundle.llm.complete(
        [{"role": "user", "content": "offline smoke"}]
    )
    vectors = bundle.embedding.embed(
        ["linux shared memory", "artifact store"]
    )

    assert isinstance(response.text, str)
    assert vectors.shape == (2, bundle.embedding.dim)


def test_factory_default_matches_config_default() -> None:
    config = AgentIPCConfig()

    assert config.llm_provider == "mock"
    assert config.embedding_provider == "hash"

    bundle = create_provider_bundle(config)

    assert isinstance(bundle.llm, MockLLMProvider)
    assert isinstance(bundle.embedding, HashEmbeddingProvider)


def test_provider_bundle_can_be_reused() -> None:
    bundle = create_provider_bundle(
        AgentIPCConfig(),
        llm_options={"default_text": "offline"},
    )

    first_response = bundle.llm.complete(
        [{"role": "user", "content": "first"}]
    )
    first_count = bundle.llm.call_count
    second_response = bundle.llm.complete(
        [{"role": "user", "content": "second"}]
    )

    first_vectors = bundle.embedding.embed(["same text"])
    second_vectors = bundle.embedding.embed(["same text"])

    assert first_response.text == "offline"
    assert second_response.text == "offline"
    assert first_count >= 1
    assert bundle.llm.call_count == first_count + 1
    np.testing.assert_array_equal(first_vectors, second_vectors)


def test_factory_does_not_modify_config() -> None:
    config = AgentIPCConfig()
    before = config.model_dump()

    create_provider_bundle(
        config,
        llm_options={"default_text": "offline"},
        embedding_options={"dim": 32},
    )

    after = config.model_dump()
    assert before == after
