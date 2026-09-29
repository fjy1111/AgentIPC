from types import SimpleNamespace

import numpy as np
import pytest

import agentipc.providers.openai_embedding as provider_module
from agentipc.providers.base import EmbeddingProvider
from agentipc.providers.openai_embedding import OpenAICompatibleEmbeddingProvider


class FakeEmbeddings:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []
        self.error: BaseException | None = None

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeOpenAI:
    def __init__(self, response: object) -> None:
        self.response = response
        self.init_calls: list[dict[str, object]] = []
        self.client: SimpleNamespace | None = None

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.init_calls.append(kwargs)
        embeddings = FakeEmbeddings(self.response)
        self.client = SimpleNamespace(embeddings=embeddings)
        return self.client


def embedding_response(
    embeddings: list[list[float]],
) -> SimpleNamespace:
    data = [
        SimpleNamespace(embedding=emb, index=idx)
        for idx, emb in enumerate(embeddings)
    ]
    return SimpleNamespace(data=data)


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    response: object | None = None,
) -> FakeOpenAI:
    fake = FakeOpenAI(response if response is not None else embedding_response([]))
    monkeypatch.setattr(provider_module, "_load_openai_class", lambda: fake)
    return fake


def test_module_import_does_not_require_openai() -> None:
    assert OpenAICompatibleEmbeddingProvider is not None


def test_missing_optional_dependency_is_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        assert name == "openai"
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    monkeypatch.setattr(provider_module.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match=r"agentipc\[openai\]"):
        OpenAICompatibleEmbeddingProvider(model="model", dim=128)


def test_constructor_forwards_options_and_satisfies_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    provider = OpenAICompatibleEmbeddingProvider(
        model=" model ",
        dim=1536,
        api_key="key",
        base_url="http://localhost:8001/v1/",
        timeout_sec=12,
    )

    assert isinstance(provider, EmbeddingProvider)
    assert provider.dim == 1536
    assert fake.init_calls == [
        {
            "api_key": "key",
            "base_url": "http://localhost:8001/v1/",
            "timeout": 12.0,
            "max_retries": 0,
        }
    ]


def test_none_optional_client_options_may_be_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    OpenAICompatibleEmbeddingProvider(model="model", dim=128)
    assert fake.init_calls == [{"timeout": 60.0, "max_retries": 0}]


@pytest.mark.parametrize("value", [1, True, False, None, []])
def test_model_type_validation(value: object, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(TypeError, match="model must be a str"):
        OpenAICompatibleEmbeddingProvider(model=value, dim=128)  # type: ignore[arg-type]


def test_empty_model_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(ValueError, match="model must be a non-empty str"):
        OpenAICompatibleEmbeddingProvider(model="", dim=128)


@pytest.mark.parametrize("value", [True, False, "128", 128.5, None, []])
def test_dim_type_validation(value: object, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(TypeError, match="dim must be an int"):
        OpenAICompatibleEmbeddingProvider(model="model", dim=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, -128])
def test_dim_value_validation(value: int, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(ValueError, match="dim must be > 0"):
        OpenAICompatibleEmbeddingProvider(model="model", dim=value)


@pytest.mark.parametrize("field", ["api_key", "base_url"])
@pytest.mark.parametrize("value", [1, True, object()])
def test_optional_string_type_validation(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch)
    with pytest.raises(TypeError):
        OpenAICompatibleEmbeddingProvider(
            model="model", dim=128, **{field: value}  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["api_key", "base_url"])
def test_optional_string_empty_value_is_rejected(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch)
    with pytest.raises(ValueError):
        OpenAICompatibleEmbeddingProvider(model="model", dim=128, **{field: ""})


@pytest.mark.parametrize("value", [True, False, "60", None])
def test_timeout_type_validation(value: object, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(TypeError, match="timeout_sec must be an int or float"):
        OpenAICompatibleEmbeddingProvider(
            model="model", dim=128, timeout_sec=value  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "value",
    [0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_timeout_value_validation(value: float, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(ValueError):
        OpenAICompatibleEmbeddingProvider(model="model", dim=128, timeout_sec=value)


@pytest.mark.parametrize("texts", ["text", (), None, ["bad", 1], [1, 2]])
def test_texts_validation(texts: object, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=128)
    with pytest.raises(TypeError):
        provider.embed(texts)  # type: ignore[arg-type]


def test_empty_input_returns_empty_array_without_api_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=128)

    result = provider.embed([])

    assert fake.client is not None
    assert fake.client.embeddings.calls == []
    assert result.shape == (0, 128)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]


def test_single_embedding_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(
        monkeypatch,
        embedding_response([[0.1, 0.2, 0.3]]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model=" model ", dim=3)

    result = provider.embed(["hello"])

    assert fake.client is not None
    assert fake.client.embeddings.calls == [
        {"model": " model ", "input": ["hello"]}
    ]
    assert result.shape == (1, 3)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]
    assert np.allclose(result[0], [0.1, 0.2, 0.3])


def test_multiple_embeddings_request(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(
        monkeypatch,
        embedding_response([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=2)

    result = provider.embed(["first", "second", "third"])

    assert fake.client is not None
    assert fake.client.embeddings.calls == [
        {"model": "model", "input": ["first", "second", "third"]}
    ]
    assert result.shape == (3, 2)
    assert result.dtype == np.float32
    assert result.flags["C_CONTIGUOUS"]
    assert np.allclose(result[0], [1.0, 0.0])
    assert np.allclose(result[1], [0.0, 1.0])
    assert np.allclose(result[2], [0.5, 0.5])


def test_api_call_kwargs_exact(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(
        monkeypatch,
        embedding_response([[0.1]]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model="test-model", dim=1)

    provider.embed(["text"])

    assert fake.client is not None
    assert len(fake.client.embeddings.calls) == 1
    call = fake.client.embeddings.calls[0]
    assert set(call.keys()) == {"model", "input"}
    assert call["model"] == "test-model"
    assert call["input"] == ["text"]


def test_output_dtype_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(
        monkeypatch,
        embedding_response([[1.0, 2.0]]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=2)

    result = provider.embed(["text"])

    assert result.dtype == np.float32


def test_output_contiguous(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(
        monkeypatch,
        embedding_response([[1.0, 2.0]]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=2)

    result = provider.embed(["text"])

    assert result.flags["C_CONTIGUOUS"]


def test_row_count_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(
        monkeypatch,
        embedding_response([[0.1]]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=1)

    with pytest.raises(ValueError, match="row count .* does not match"):
        provider.embed(["first", "second"])


def test_dimension_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(
        monkeypatch,
        embedding_response([[0.1, 0.2]]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=3)

    with pytest.raises(ValueError, match="has dimension 2, expected 3"):
        provider.embed(["text"])


def test_non_1d_embedding_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(monkeypatch)
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=2)
    # Fake response with 2D embedding
    fake.client.embeddings.response = SimpleNamespace(
        data=[SimpleNamespace(embedding=[[0.1, 0.2]], index=0)]
    )

    with pytest.raises(ValueError, match="must be 1-dimensional"):
        provider.embed(["text"])


def test_nan_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(
        monkeypatch,
        embedding_response([[float("nan"), 0.2]]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=2)

    with pytest.raises(ValueError, match="non-finite values"):
        provider.embed(["text"])


def test_inf_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(
        monkeypatch,
        embedding_response([[float("inf"), 0.2]]),
    )
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=2)

    with pytest.raises(ValueError, match="non-finite values"):
        provider.embed(["text"])


def test_non_numeric_embedding_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(monkeypatch)
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=2)
    # Fake response with non-numeric embedding
    fake.client.embeddings.response = SimpleNamespace(
        data=[SimpleNamespace(embedding=["a", "b"], index=0)]
    )

    with pytest.raises((ValueError, TypeError)):
        provider.embed(["text"])


def test_sdk_exception_identity_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(monkeypatch)
    provider = OpenAICompatibleEmbeddingProvider(model="model", dim=128)
    assert fake.client is not None
    error = RuntimeError("sdk boom")
    fake.client.embeddings.error = error

    with pytest.raises(RuntimeError) as exc_info:
        provider.embed(["text"])

    assert exc_info.value is error
