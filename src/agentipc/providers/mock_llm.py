from collections.abc import Mapping
import math

from agentipc.providers.base import LLMResponse


class MockLLMProvider:
    def __init__(
        self,
        *,
        keyword_responses: Mapping[str, str] | None = None,
        default_text: str = "mock response",
    ) -> None:
        if keyword_responses is not None and not isinstance(
            keyword_responses,
            Mapping,
        ):
            raise TypeError("keyword_responses must be a Mapping[str, str] or None")
        if not isinstance(default_text, str):
            raise TypeError("default_text must be a str")

        entries: list[tuple[str, str, str]] = []
        seen_casefolded: set[str] = set()
        if keyword_responses is not None:
            for keyword, response in keyword_responses.items():
                if not isinstance(keyword, str):
                    raise TypeError("keyword_responses keys must be str")
                if keyword == "":
                    raise ValueError("keyword_responses keys must be non-empty str")
                if not isinstance(response, str):
                    raise TypeError("keyword_responses values must be str")

                folded = keyword.casefold()
                if folded in seen_casefolded:
                    raise ValueError("keyword_responses contains casefold-duplicate keys")
                seen_casefolded.add(folded)
                entries.append((keyword, folded, response))

        self._keyword_responses = tuple(
            sorted(
                entries,
                key=lambda item: (-len(item[0]), item[1]),
            )
        )
        self._default_text = default_text
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._validate_messages(messages)
        self._validate_temperature(temperature)

        matched_keyword: str | None = None
        text = self._default_text

        searchable_contents = [
            message["content"].casefold()
            for message in messages
            if "content" in message
        ]
        for keyword, folded_keyword, response in self._keyword_responses:
            if any(folded_keyword in content for content in searchable_contents):
                matched_keyword = keyword
                text = response
                break

        response = LLMResponse(
            text=text,
            prompt_tokens=None,
            completion_tokens=None,
            latency_ms=0.0,
            raw={
                "provider": "mock",
                "matched_keyword": matched_keyword,
            },
        )
        self._call_count += 1
        return response

    @staticmethod
    def _validate_messages(messages: list[dict[str, str]]) -> None:
        if not isinstance(messages, list):
            raise TypeError("messages must be a list[dict[str, str]]")

        for message in messages:
            if not isinstance(message, dict):
                raise TypeError("each message must be a dict[str, str]")
            if not all(isinstance(key, str) for key in message):
                raise TypeError("message keys must be str")
            if not all(isinstance(value, str) for value in message.values()):
                raise TypeError("message values must be str")

    @staticmethod
    def _validate_temperature(temperature: float) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be an int or float")
        if not math.isfinite(temperature):
            raise ValueError("temperature must be finite")
        if temperature < 0:
            raise ValueError("temperature must be >= 0")
