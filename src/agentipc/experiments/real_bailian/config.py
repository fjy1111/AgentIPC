from __future__ import annotations

import math
import os
import re
from collections.abc import Mapping


DEFAULT_LLM_MODEL = "qwen3.7-plus-2026-05-26"
DEFAULT_EMBEDDING_MODEL = "qwen3.7-text-embedding"
DEFAULT_EMBEDDING_DIM = 1024
DEFAULT_TEMPERATURE = 0.0

_API_KEY_ENV = "DASHSCOPE_API_KEY"
_BASE_URL_ENV = "AGENTIPC_BAILIAN_BASE_URL"
_REGION_ENV = "AGENTIPC_BAILIAN_REGION"


class RealBailianConfigError(ValueError):
    """Raised for safe-to-display real experiment configuration errors."""


class RealBailianConfig:
    """Secret-safe configuration for the paid Bailian calibration path.

    The API key and full base URL are intentionally stored in private slots and
    omitted from ``repr`` and all public serialization helpers.
    """

    __slots__ = (
        "_api_key",
        "_base_url",
        "region",
        "llm_model",
        "embedding_model",
        "embedding_dim",
        "temperature",
    )

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        region: str,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self._api_key = _require_non_empty_str(api_key, "api_key")
        self._base_url = _require_non_empty_str(base_url, "base_url")
        self.region = _require_region_label(region)
        self.llm_model = _require_non_empty_str(llm_model, "llm_model")
        self.embedding_model = _require_non_empty_str(
            embedding_model,
            "embedding_model",
        )
        if type(embedding_dim) is not int:
            raise TypeError("embedding_dim must be an int")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be > 0")
        if isinstance(temperature, bool) or not isinstance(
            temperature,
            (int, float),
        ):
            raise TypeError("temperature must be an int or float")
        normalized_temperature = float(temperature)
        if not math.isfinite(normalized_temperature):
            raise ValueError("temperature must be finite")
        if normalized_temperature < 0.0:
            raise ValueError("temperature must be >= 0")

        self.embedding_dim = embedding_dim
        self.temperature = normalized_temperature

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    def public_fields(self) -> dict[str, object]:
        """Return only fields that are safe to persist in experiment outputs."""
        return {
            "api_region": self.region,
            "llm_provider": "openai_compatible",
            "llm_model": self.llm_model,
            "temperature": self.temperature,
            "embedding_provider": "openai_compatible",
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }

    def __repr__(self) -> str:
        return (
            "RealBailianConfig("
            f"region={self.region!r}, "
            f"llm_model={self.llm_model!r}, "
            f"embedding_model={self.embedding_model!r}, "
            f"embedding_dim={self.embedding_dim!r}, "
            f"temperature={self.temperature!r}, "
            "credentials=<redacted>)"
        )


def _require_non_empty_str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty str")
    return value


def _require_region_label(value: object) -> str:
    region = _require_non_empty_str(value, "region").strip()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", region) is None:
        raise ValueError(
            "region must be a short label such as beijing, singapore, or hongkong"
        )
    return region


def load_real_bailian_config(
    environ: Mapping[str, str] | None = None,
) -> RealBailianConfig:
    """Load only the three explicitly allowed environment variables.

    This function must be called *after* the ``--confirm-real-api`` gate.
    """
    env = os.environ if environ is None else environ

    api_key = env.get(_API_KEY_ENV)
    if not api_key:
        raise RealBailianConfigError(f"{_API_KEY_ENV} is required")

    base_url = env.get(_BASE_URL_ENV)
    if not base_url:
        raise RealBailianConfigError(f"{_BASE_URL_ENV} is required")

    region = env.get(_REGION_ENV)
    if not region:
        raise RealBailianConfigError(f"{_REGION_ENV} is required")

    return RealBailianConfig(
        api_key=api_key,
        base_url=base_url,
        region=region,
    )
