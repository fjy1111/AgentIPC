import importlib
import math
import time
from typing import Any

from agentipc.providers.base import LLMResponse


def _load_openai_class() -> type[Any]:
    try:
        module = importlib.import_module("openai")
    except ModuleNotFoundError as exc:
        if exc.name != "openai":
            raise
        raise RuntimeError(
            "OpenAI-compatible provider requires optional dependency; "
            "install agentipc[openai]"
        ) from exc
    return module.OpenAI


class OpenAICompatibleProvider:
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_sec: float = 60.0,
    ) -> None:
        if not isinstance(model, str):
            raise TypeError("model must be a str")
        if model == "":
            raise ValueError("model must be a non-empty str")

        if api_key is not None and not isinstance(api_key, str):
            raise TypeError("api_key must be a str or None")
        if api_key == "":
            raise ValueError("api_key must be a non-empty str or None")

        if base_url is not None and not isinstance(base_url, str):
            raise TypeError("base_url must be a str or None")
        if base_url == "":
            raise ValueError("base_url must be a non-empty str or None")

        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
            raise TypeError("timeout_sec must be an int or float")
        if not math.isfinite(timeout_sec):
            raise ValueError("timeout_sec must be finite")
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be > 0")

        self._model = model

        openai_class = _load_openai_class()
        client_kwargs: dict[str, Any] = {
            "timeout": float(timeout_sec),
            "max_retries": 0,
        }
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self._client = openai_class(**client_kwargs)

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self._validate_messages(messages)
        self._validate_temperature(temperature)

        started_at = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=float(temperature),
        )
        finished_at = time.perf_counter()
        latency_ms = max(0.0, (finished_at - started_at) * 1000.0)

        choices = completion.choices
        if not choices:
            raise ValueError("completion.choices must contain at least one choice")

        choice = choices[0]
        content = choice.message.content
        if content is None:
            text = ""
        elif isinstance(content, str):
            text = content
        else:
            raise TypeError("completion.choices[0].message.content must be str or None")

        usage = getattr(completion, "usage", None)
        if usage is None:
            prompt_tokens = None
            completion_tokens = None
        else:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens

        request_id = getattr(completion, "_request_id", None)
        if request_id is not None and not isinstance(request_id, str):
            request_id = str(request_id)

        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason is not None and not isinstance(finish_reason, str):
            finish_reason = str(finish_reason)

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            raw={
                "provider": "openai_compatible",
                "model": self._model,
                "request_id": request_id,
                "finish_reason": finish_reason,
            },
        )

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
