from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agentipc.evaluation.text_counter import TextCount, TextCounter


def test_exact_chars_without_tiktoken() -> None:
    result = TextCounter(use_tiktoken=False).count("A中🙂\n")

    assert result.text_chars == 4
    assert result.text_tokens == 0
    assert result.token_method == "unavailable"


def test_empty_text() -> None:
    result = TextCounter(use_tiktoken=False).count("")

    assert result.text_chars == 0
    assert result.text_tokens == 0
    assert result.token_method == "unavailable"


def test_disabled_tiktoken_never_loads_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_loader() -> object:
        raise AssertionError("tiktoken loader must not be called")

    monkeypatch.setattr("agentipc.evaluation.text_counter._load_tiktoken", fail_loader)

    result = TextCounter(use_tiktoken=False).count("hello")

    assert result.text_chars == 5
    assert result.text_tokens == 0
    assert result.token_method == "unavailable"


def test_missing_tiktoken_keeps_exact_char_count(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_loader() -> object:
        raise ModuleNotFoundError("No module named 'tiktoken'")

    monkeypatch.setattr("agentipc.evaluation.text_counter._load_tiktoken", missing_loader)

    result = TextCounter().count("A中🙂\n")

    assert result.text_chars == 4
    assert result.text_tokens == 0
    assert result.token_method == "unavailable"


def test_fake_tiktoken_counts_tokens_and_records_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeEncoding:
        def encode(self, text: str) -> list[int]:
            assert text == "hello world"
            return [10, 20, 30]

    def get_encoding(name: str) -> FakeEncoding:
        calls.append(name)
        return FakeEncoding()

    monkeypatch.setattr(
        "agentipc.evaluation.text_counter._load_tiktoken",
        lambda: SimpleNamespace(get_encoding=get_encoding),
    )

    result = TextCounter(encoding_name="test_encoding").count("hello world")

    assert result.text_chars == 11
    assert result.text_tokens == 3
    assert result.token_method == "tiktoken:test_encoding"
    assert calls == ["test_encoding"]


def test_encoding_is_cached_per_counter_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class FakeEncoding:
        def encode(self, text: str) -> list[int]:
            return list(range(len(text)))

    def get_encoding(name: str) -> FakeEncoding:
        calls.append(name)
        return FakeEncoding()

    monkeypatch.setattr(
        "agentipc.evaluation.text_counter._load_tiktoken",
        lambda: SimpleNamespace(get_encoding=get_encoding),
    )
    counter = TextCounter()

    assert counter.count("a").text_tokens == 1
    assert counter.count("ab").text_tokens == 2
    assert calls == ["cl100k_base"]


def test_tokenizer_get_encoding_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def get_encoding(name: str) -> object:
        raise RuntimeError(f"bad encoding: {name}")

    monkeypatch.setattr(
        "agentipc.evaluation.text_counter._load_tiktoken",
        lambda: SimpleNamespace(get_encoding=get_encoding),
    )

    with pytest.raises(RuntimeError, match="bad encoding"):
        TextCounter().count("hello")


def test_tokenizer_encode_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEncoding:
        def encode(self, text: str) -> list[int]:
            raise RuntimeError("encode failed")

    monkeypatch.setattr(
        "agentipc.evaluation.text_counter._load_tiktoken",
        lambda: SimpleNamespace(get_encoding=lambda name: FakeEncoding()),
    )

    with pytest.raises(RuntimeError, match="encode failed"):
        TextCounter().count("hello")


@pytest.mark.parametrize("value", [None, b"text", 1, []])
def test_count_rejects_non_string_input(value: object) -> None:
    with pytest.raises(TypeError):
        TextCounter(use_tiktoken=False).count(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, 1, b"cl100k_base"])
def test_encoding_name_rejects_non_string(value: object) -> None:
    with pytest.raises(TypeError):
        TextCounter(encoding_name=value)  # type: ignore[arg-type]


def test_encoding_name_rejects_empty_string() -> None:
    with pytest.raises(ValueError):
        TextCounter(encoding_name="")


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_use_tiktoken_requires_exact_bool(value: object) -> None:
    with pytest.raises(TypeError):
        TextCounter(use_tiktoken=value)  # type: ignore[arg-type]


def test_text_count_forbids_extra_fields_and_requires_strict_non_negative_ints() -> None:
    with pytest.raises(ValidationError):
        TextCount(text_chars=True, text_tokens=0, token_method="unavailable")
    with pytest.raises(ValidationError):
        TextCount(text_chars=0, text_tokens=-1, token_method="unavailable")
    with pytest.raises(ValidationError):
        TextCount(text_chars=0, text_tokens=0, token_method="")
    with pytest.raises(ValidationError):
        TextCount(text_chars=0, text_tokens=0, token_method="unavailable", extra=1)
