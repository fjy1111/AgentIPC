from __future__ import annotations

import importlib
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field


StrictNonNegativeInt = Annotated[int, Field(ge=0, strict=True)]


class TextCount(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text_chars: StrictNonNegativeInt
    text_tokens: StrictNonNegativeInt
    token_method: str = Field(min_length=1)


def _load_tiktoken() -> Any:
    return importlib.import_module("tiktoken")


class TextCounter:
    def __init__(
        self,
        *,
        encoding_name: str = "cl100k_base",
        use_tiktoken: bool = True,
    ) -> None:
        if not isinstance(encoding_name, str):
            raise TypeError("encoding_name must be a string")
        if encoding_name == "":
            raise ValueError("encoding_name must be non-empty")
        if type(use_tiktoken) is not bool:
            raise TypeError("use_tiktoken must be a bool")

        self._encoding_name = encoding_name
        self._use_tiktoken = use_tiktoken
        self._encoding: Any | None = None

    def count(self, text: str) -> TextCount:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text_chars = len(text)
        if not self._use_tiktoken:
            return TextCount(
                text_chars=text_chars,
                text_tokens=0,
                token_method="unavailable",
            )

        if self._encoding is None:
            try:
                tiktoken = _load_tiktoken()
            except (ModuleNotFoundError, ImportError):
                return TextCount(
                    text_chars=text_chars,
                    text_tokens=0,
                    token_method="unavailable",
                )
            self._encoding = tiktoken.get_encoding(self._encoding_name)

        return TextCount(
            text_chars=text_chars,
            text_tokens=len(self._encoding.encode(text)),
            token_method=f"tiktoken:{self._encoding_name}",
        )
