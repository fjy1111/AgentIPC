import numpy as np
import pytest

import agentipc.providers.sentence_transformer as provider_module
from agentipc.providers.base import EmbeddingProvider
from agentipc.providers.sentence_transformer import SentenceTransformerEmbeddingProvider


class FakeModel:
    def __init__(self, dimension: object = 3, output: object | None = None) -> None:
        self.dimension = dimension
        self.output = output
        self.dimension_calls = 0
        self.encode_calls: list[tuple[list[str], dict[str, object]]] = []
        self.encode_error: BaseException | None = None

    def get_embedding_dimension(self) -> object:
        self.dimension_calls += 1
        return self.dimension

    def encode(self, texts: list[str], **kwargs: object) -> object:
        self.encode_calls.append((texts, kwargs))
        if self.encode_error is not None:
            raise self.encode_error
        if self.output is not None:
            return self.output
        return np.arange(len(texts) * int(self.dimension), dtype=np.float64).reshape(
            len(texts), int(self.dimension)
        )


class FakeSentenceTransformer:
    def __init__(self, model: FakeModel | None = None) -> None:
        self.model = model or FakeModel()
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.init_error: BaseException | None = None

    def __call__(self, model_name: str, **kwargs: object) -> FakeModel:
        self.calls.append((model_name, kwargs))
        if self.init_error is not None:
            raise self.init_error
        return self.model


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    model: FakeModel | None = None,
) -> FakeSentenceTransformer:
    fake = FakeSentenceTransformer(model)
    monkeypatch.setattr(
        provider_module,
        "_load_sentence_transformer_class",
        lambda: fake,
    )
    return fake


def test_module_import_and_constructor_are_lazy(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def fail_if_loaded() -> object:
        nonlocal called
        called = True
        raise AssertionError("constructor must not load sentence-transformers")

    monkeypatch.setattr(provider_module, "_load_sentence_transformer_class", fail_if_loaded)
    provider = SentenceTransformerEmbeddingProvider("model")

    assert SentenceTransformerEmbeddingProvider is not None
    assert isinstance(provider, EmbeddingProvider)
    assert called is False


def test_first_dim_loads_once_forwards_options_and_caches_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch, FakeModel(dimension=7))
    provider = SentenceTransformerEmbeddingProvider(" model ", device="cpu")

    assert provider.dim == 7
    assert provider.dim == 7
    assert fake.calls == [
        (" model ", {"local_files_only": True, "device": "cpu"})
    ]
    assert fake.model.dimension_calls == 1


def test_first_embed_and_repeated_calls_reuse_one_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    model = FakeModel(dimension=3, output=source)
    fake = install_fake(monkeypatch, model)
    provider = SentenceTransformerEmbeddingProvider("model")

    first = provider.embed(["a", "b"])
    second = provider.embed(["c", "d"])
    assert provider.dim == 3

    assert len(fake.calls) == 1
    assert model.dimension_calls == 1
    assert model.encode_calls[0] == (
        ["a", "b"],
        {
            "convert_to_numpy": True,
            "convert_to_tensor": False,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        },
    )
    for result in (first, second):
        assert result.shape == (2, 3)
        assert result.dtype == np.float32
        assert result.flags.c_contiguous
        assert np.all(np.isfinite(result))


def test_local_files_only_false_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(monkeypatch)
    provider = SentenceTransformerEmbeddingProvider("model", local_files_only=False)
    _ = provider.dim
    assert fake.calls == [("model", {"local_files_only": False})]


def test_device_none_may_be_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(monkeypatch)
    provider = SentenceTransformerEmbeddingProvider("model")
    _ = provider.dim
    assert fake.calls == [("model", {"local_files_only": True})]


@pytest.mark.parametrize("value", [1, True, None])
def test_model_name_type_validation(value: object) -> None:
    with pytest.raises(TypeError):
        SentenceTransformerEmbeddingProvider(value)  # type: ignore[arg-type]


def test_empty_model_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddingProvider("")


@pytest.mark.parametrize("value", [1, True, object()])
def test_device_type_validation(value: object) -> None:
    with pytest.raises(TypeError):
        SentenceTransformerEmbeddingProvider("model", device=value)  # type: ignore[arg-type]


def test_empty_device_is_rejected() -> None:
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddingProvider("model", device="")


@pytest.mark.parametrize("value", [0, 1, "true", None])
def test_local_files_only_requires_exact_bool(value: object) -> None:
    with pytest.raises(TypeError):
        SentenceTransformerEmbeddingProvider(
            "model",
            local_files_only=value,  # type: ignore[arg-type]
        )


def test_missing_optional_dependency_is_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        assert name == "sentence_transformers"
        raise ModuleNotFoundError(
            "No module named 'sentence_transformers'",
            name="sentence_transformers",
        )

    monkeypatch.setattr(provider_module.importlib, "import_module", missing)
    provider = SentenceTransformerEmbeddingProvider("model")

    with pytest.raises(RuntimeError, match=r"agentipc\[sentence-transformers\]"):
        _ = provider.dim


@pytest.mark.parametrize("dimension", [None, True, False, 0, -1, 3.0, "3"])
def test_invalid_model_dimension_is_rejected(
    dimension: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch, FakeModel(dimension=dimension))
    with pytest.raises(ValueError):
        _ = SentenceTransformerEmbeddingProvider("model").dim


def test_empty_batch_returns_float32_c_contiguous_without_encode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(dimension=5)
    install_fake(monkeypatch, model)
    result = SentenceTransformerEmbeddingProvider("model").embed([])

    assert result.shape == (0, 5)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    assert model.encode_calls == []


@pytest.mark.parametrize("texts", [("a",), "a", None, ["ok", 1]])
def test_embed_input_validation(texts: object) -> None:
    provider = SentenceTransformerEmbeddingProvider("model")
    with pytest.raises(TypeError):
        provider.embed(texts)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("output", "texts"),
    [
        (np.zeros((1, 3), dtype=np.float32), ["a", "b"]),
        (np.zeros((2, 4), dtype=np.float32), ["a", "b"]),
        (np.zeros(3, dtype=np.float32), ["a"]),
        (np.array([[0.0, float("nan"), 0.0]], dtype=np.float32), ["a"]),
        (np.array([[0.0, float("inf"), 0.0]], dtype=np.float32), ["a"]),
    ],
)
def test_invalid_encode_output_is_rejected(
    output: np.ndarray,
    texts: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch, FakeModel(dimension=3, output=output))
    with pytest.raises(ValueError):
        SentenceTransformerEmbeddingProvider("model").embed(texts)


def test_model_initialization_exception_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    error = RuntimeError("model init boom")
    fake.init_error = error

    with pytest.raises(RuntimeError) as exc_info:
        _ = SentenceTransformerEmbeddingProvider("model").dim

    assert exc_info.value is error


def test_encode_exception_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = FakeModel(dimension=3)
    error = RuntimeError("encode boom")
    model.encode_error = error
    install_fake(monkeypatch, model)

    with pytest.raises(RuntimeError) as exc_info:
        SentenceTransformerEmbeddingProvider("model").embed(["a"])

    assert exc_info.value is error
