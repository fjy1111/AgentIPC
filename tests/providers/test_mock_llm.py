from collections import OrderedDict

import pytest

from agentipc.providers.base import LLMResponse
from agentipc.providers.mock_llm import MockLLMProvider


def test_default_provider_returns_mock_response() -> None:
    provider = MockLLMProvider()

    response = provider.complete([{"role": "user", "content": "hello"}])

    assert response.text == "mock response"


def test_custom_default_text_is_returned() -> None:
    provider = MockLLMProvider(default_text="custom")

    assert provider.complete([]).text == "custom"


def test_keyword_match_is_case_insensitive() -> None:
    provider = MockLLMProvider(keyword_responses={"plan": "planned"})

    response = provider.complete([{"content": "Please create a PLAN"}])

    assert response.text == "planned"
    assert response.raw == {"provider": "mock", "matched_keyword": "plan"}


def test_longest_keyword_wins() -> None:
    provider = MockLLMProvider(
        keyword_responses={"log": "short", "log analysis": "long"}
    )

    response = provider.complete([{"content": "run LOG ANALYSIS now"}])

    assert response.text == "long"
    assert response.raw["matched_keyword"] == "log analysis"  # type: ignore[index]


def test_same_length_tie_uses_casefolded_keyword_order() -> None:
    provider = MockLLMProvider(
        keyword_responses={"zeta": "second", "beta": "first"}
    )

    response = provider.complete([{"content": "zeta beta"}])

    assert response.text == "first"
    assert response.raw["matched_keyword"] == "beta"  # type: ignore[index]


def test_insertion_order_does_not_change_keyword_selection() -> None:
    first = MockLLMProvider(
        keyword_responses=OrderedDict(
            [("zeta", "second"), ("beta", "first")]
        )
    )
    second = MockLLMProvider(
        keyword_responses=OrderedDict(
            [("beta", "first"), ("zeta", "second")]
        )
    )
    messages = [{"content": "zeta beta"}]

    assert first.complete(messages).text == "first"
    assert second.complete(messages).text == "first"


def test_no_keyword_uses_default() -> None:
    provider = MockLLMProvider(
        keyword_responses={"plan": "planned"},
        default_text="fallback",
    )

    response = provider.complete([{"content": "summarize"}])

    assert response.text == "fallback"
    assert response.raw["matched_keyword"] is None  # type: ignore[index]


def test_multiple_messages_search_content() -> None:
    provider = MockLLMProvider(keyword_responses={"summary": "done"})

    response = provider.complete(
        [
            {"role": "system", "content": "prepare"},
            {"role": "user", "content": "write SUMMARY"},
        ]
    )

    assert response.text == "done"


def test_role_and_other_fields_do_not_trigger_keyword() -> None:
    provider = MockLLMProvider(
        keyword_responses={"plan": "matched"},
        default_text="default",
    )

    response = provider.complete(
        [
            {"role": "PLAN", "metadata": "plan"},
            {"content": "nothing relevant"},
        ]
    )

    assert response.text == "default"


def test_empty_messages_are_allowed() -> None:
    provider = MockLLMProvider(default_text="empty")

    assert provider.complete([]).text == "empty"


def test_call_count_starts_at_zero_and_successful_calls_increment() -> None:
    provider = MockLLMProvider()

    assert provider.call_count == 0
    provider.complete([])
    assert provider.call_count == 1
    provider.complete([])
    assert provider.call_count == 2


def test_call_counts_are_instance_local() -> None:
    first = MockLLMProvider()
    second = MockLLMProvider()

    first.complete([])
    first.complete([])
    second.complete([])

    assert first.call_count == 2
    assert second.call_count == 1


def test_invalid_call_does_not_increment_call_count() -> None:
    provider = MockLLMProvider()

    with pytest.raises(TypeError):
        provider.complete("bad")  # type: ignore[arg-type]

    assert provider.call_count == 0


def test_configured_mapping_is_copied() -> None:
    mapping = {"plan": "first"}
    provider = MockLLMProvider(keyword_responses=mapping)
    mapping["plan"] = "changed"
    mapping["summary"] = "new"

    assert provider.complete([{"content": "plan"}]).text == "first"
    assert provider.complete([{"content": "summary"}]).text == "mock response"


def test_invalid_mapping_type_is_rejected() -> None:
    with pytest.raises(TypeError):
        MockLLMProvider(keyword_responses=[("plan", "x")])  # type: ignore[arg-type]


def test_empty_keyword_is_rejected() -> None:
    with pytest.raises(ValueError):
        MockLLMProvider(keyword_responses={"": "x"})


def test_non_string_keyword_is_rejected() -> None:
    with pytest.raises(TypeError):
        MockLLMProvider(keyword_responses={1: "x"})  # type: ignore[dict-item]


def test_non_string_response_is_rejected() -> None:
    with pytest.raises(TypeError):
        MockLLMProvider(keyword_responses={"plan": 1})  # type: ignore[dict-item]


def test_casefold_duplicate_keyword_is_rejected() -> None:
    with pytest.raises(ValueError):
        MockLLMProvider(keyword_responses={"PLAN": "x", "plan": "y"})


def test_unicode_casefold_duplicate_keyword_is_rejected() -> None:
    with pytest.raises(ValueError):
        MockLLMProvider(keyword_responses={"ß": "x", "ss": "y"})


def test_non_string_default_text_is_rejected() -> None:
    with pytest.raises(TypeError):
        MockLLMProvider(default_text=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("messages", ["text", (), iter([]), None])
def test_messages_non_list_is_rejected(messages: object) -> None:
    provider = MockLLMProvider()

    with pytest.raises(TypeError):
        provider.complete(messages)  # type: ignore[arg-type]


def test_message_non_dict_is_rejected() -> None:
    provider = MockLLMProvider()

    with pytest.raises(TypeError):
        provider.complete(["bad"])  # type: ignore[list-item]


def test_non_string_message_key_is_rejected() -> None:
    provider = MockLLMProvider()

    with pytest.raises(TypeError):
        provider.complete([{1: "value"}])  # type: ignore[dict-item]


def test_non_string_message_value_is_rejected() -> None:
    provider = MockLLMProvider()

    with pytest.raises(TypeError):
        provider.complete([{"content": 1}])  # type: ignore[dict-item]


@pytest.mark.parametrize("temperature", [-0.1, float("nan"), float("inf"), float("-inf")])
def test_invalid_numeric_temperature_is_rejected(temperature: float) -> None:
    provider = MockLLMProvider()

    with pytest.raises(ValueError):
        provider.complete([], temperature=temperature)


def test_bool_temperature_is_rejected() -> None:
    provider = MockLLMProvider()

    with pytest.raises(TypeError):
        provider.complete([], temperature=True)


def test_non_numeric_temperature_is_rejected() -> None:
    provider = MockLLMProvider()

    with pytest.raises(TypeError):
        provider.complete([], temperature="0")  # type: ignore[arg-type]


def test_valid_temperature_does_not_change_output() -> None:
    provider = MockLLMProvider(keyword_responses={"plan": "same"})
    messages = [{"content": "plan"}]

    assert provider.complete(messages, temperature=0).text == "same"
    assert provider.complete(messages, temperature=99.5).text == "same"


def test_response_contract_is_fixed() -> None:
    provider = MockLLMProvider(keyword_responses={"plan": "planned"})

    response = provider.complete([{"content": "PLAN"}])

    assert isinstance(response, LLMResponse)
    assert response.text == "planned"
    assert response.prompt_tokens is None
    assert response.completion_tokens is None
    assert response.latency_ms == 0.0
    assert response.raw == {
        "provider": "mock",
        "matched_keyword": "plan",
    }
