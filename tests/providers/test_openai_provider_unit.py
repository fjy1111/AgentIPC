from types import SimpleNamespace

import pytest

import agentipc.providers.openai_compatible as provider_module
from agentipc.providers.base import LLMProvider, LLMResponse
from agentipc.providers.openai_compatible import OpenAICompatibleProvider


class FakeCompletions:
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
        completions = FakeCompletions(self.response)
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
            completions=completions,
        )
        return self.client


def completion(
    *,
    content: object = "answer",
    usage: object = None,
    request_id: object = "req-1",
    finish_reason: object = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=usage,
        _request_id=request_id,
    )


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    response: object | None = None,
) -> FakeOpenAI:
    fake = FakeOpenAI(response if response is not None else completion())
    monkeypatch.setattr(provider_module, "_load_openai_class", lambda: fake)
    return fake


def test_module_import_does_not_require_openai() -> None:
    assert OpenAICompatibleProvider is not None


def test_missing_optional_dependency_is_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(name: str) -> object:
        assert name == "openai"
        raise ModuleNotFoundError("No module named 'openai'", name="openai")

    monkeypatch.setattr(provider_module.importlib, "import_module", missing)

    with pytest.raises(RuntimeError, match=r"agentipc\[openai\]"):
        OpenAICompatibleProvider(model="model")


def test_constructor_forwards_options_and_satisfies_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    provider = OpenAICompatibleProvider(
        model=" model ",
        api_key="key",
        base_url="http://localhost:8000/v1/",
        timeout_sec=12,
    )

    assert isinstance(provider, LLMProvider)
    assert fake.init_calls == [
        {
            "api_key": "key",
            "base_url": "http://localhost:8000/v1/",
            "timeout": 12.0,
            "max_retries": 0,
        }
    ]


def test_none_optional_client_options_may_be_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(monkeypatch)
    OpenAICompatibleProvider(model="model")
    assert fake.init_calls == [{"timeout": 60.0, "max_retries": 0}]


@pytest.mark.parametrize("value", [1, True, None])
def test_model_type_validation(value: object, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(TypeError):
        OpenAICompatibleProvider(model=value)  # type: ignore[arg-type]


def test_empty_model_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(model="")


@pytest.mark.parametrize("field", ["api_key", "base_url"])
@pytest.mark.parametrize("value", [1, True, object()])
def test_optional_string_type_validation(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch)
    with pytest.raises(TypeError):
        OpenAICompatibleProvider(model="model", **{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["api_key", "base_url"])
def test_optional_string_empty_value_is_rejected(
    field: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch)
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(model="model", **{field: ""})


@pytest.mark.parametrize("value", [True, False, "60", None])
def test_timeout_type_validation(value: object, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(TypeError):
        OpenAICompatibleProvider(model="model", timeout_sec=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [0, -1, float("nan"), float("inf"), float("-inf")],
)
def test_timeout_value_validation(value: float, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(model="model", timeout_sec=value)


@pytest.mark.parametrize(
    "messages",
    ["text", (), None, ["bad"], [{1: "x"}], [{"content": 1}]],
)
def test_messages_validation(messages: object, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    provider = OpenAICompatibleProvider(model="model")
    with pytest.raises(TypeError):
        provider.complete(messages)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [True, "0", None])
def test_temperature_type_validation(value: object, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    provider = OpenAICompatibleProvider(model="model")
    with pytest.raises(TypeError):
        provider.complete([], temperature=value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [-0.1, float("nan"), float("inf"), float("-inf")],
)
def test_temperature_value_validation(value: float, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch)
    provider = OpenAICompatibleProvider(model="model")
    with pytest.raises(ValueError):
        provider.complete([], temperature=value)


def test_request_response_usage_latency_and_raw_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = install_fake(
        monkeypatch,
        completion(
            content="answer",
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
            request_id=123,
            finish_reason=456,
        ),
    )
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(provider_module.time, "perf_counter", lambda: next(ticks))
    provider = OpenAICompatibleProvider(model=" model ")
    messages = [{"role": "user", "content": "hello", "extra": "kept"}]

    response = provider.complete(messages, temperature=99.5)

    assert fake.client is not None
    assert fake.client.completions.calls == [
        {"model": " model ", "messages": messages, "temperature": 99.5}
    ]
    assert isinstance(response, LLMResponse)
    assert response.text == "answer"
    assert response.prompt_tokens == 7
    assert response.completion_tokens == 3
    assert response.latency_ms == pytest.approx(250.0)
    assert response.raw == {
        "provider": "openai_compatible",
        "model": " model ",
        "request_id": "123",
        "finish_reason": "456",
    }


def test_none_content_and_absent_usage_are_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake(monkeypatch, completion(content=None, usage=None))
    response = OpenAICompatibleProvider(model="model").complete([])
    assert response.text == ""
    assert response.prompt_tokens is None
    assert response.completion_tokens is None


def test_missing_request_id_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    choice = SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")
    install_fake(monkeypatch, SimpleNamespace(choices=[choice], usage=None))
    response = OpenAICompatibleProvider(model="model").complete([])
    assert response.raw is not None
    assert response.raw["request_id"] is None


def test_empty_choices_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, SimpleNamespace(choices=[], usage=None, _request_id=None))
    with pytest.raises(ValueError, match="choices"):
        OpenAICompatibleProvider(model="model").complete([])


def test_non_string_content_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake(monkeypatch, completion(content=["part"]))
    with pytest.raises(TypeError):
        OpenAICompatibleProvider(model="model").complete([])


def test_sdk_exception_propagates_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install_fake(monkeypatch)
    provider = OpenAICompatibleProvider(model="model")
    assert fake.client is not None
    error = RuntimeError("sdk boom")
    fake.client.completions.error = error

    with pytest.raises(RuntimeError) as exc_info:
        provider.complete([])

    assert exc_info.value is error
