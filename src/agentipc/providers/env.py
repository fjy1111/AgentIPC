"""Environment-driven provider configuration.

This module provides unified configuration of cloud and local OpenAI-compatible
providers via environment variables, without deployment-specific logic.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from agentipc.config import AgentIPCConfig
from agentipc.providers.factory import ProviderBundle, create_provider_bundle


@dataclass(frozen=True)
class ResolvedProviderBundle:
    """Bundle with both resolved config and constructed providers.

    The config reflects environment overrides but does not contain secrets.
    """

    config: AgentIPCConfig
    bundle: ProviderBundle


def create_provider_bundle_from_env(
    config: AgentIPCConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ResolvedProviderBundle:
    """Create providers from config with environment variable overrides.

    Args:
        config: Base configuration. If None, uses AgentIPCConfig() defaults.
        environ: Environment mapping. If None, uses os.environ.

    Returns:
        ResolvedProviderBundle with resolved config and constructed providers.

    Raises:
        TypeError: Invalid argument types or environ structure.
        ValueError: Invalid environment variable values.

    Environment variables:
        AGENTIPC_LLM_PROVIDER: Override config.llm_provider
        AGENTIPC_EMBEDDING_PROVIDER: Override config.embedding_provider

        For llm_provider="openai":
            AGENTIPC_LLM_MODEL: Required model name
            AGENTIPC_LLM_API_KEY: Optional API key (may be omitted for local)
            AGENTIPC_LLM_BASE_URL: Optional base URL
            AGENTIPC_LLM_TIMEOUT_SEC: Optional timeout in seconds

        For embedding_provider="hash":
            AGENTIPC_EMBEDDING_DIM: Optional dimension (int)

        For embedding_provider="sentence-transformer":
            AGENTIPC_EMBEDDING_MODEL_NAME: Required model name
            AGENTIPC_EMBEDDING_DEVICE: Optional device
            AGENTIPC_EMBEDDING_LOCAL_FILES_ONLY: Optional bool ("true"/"false")
    """
    if config is None:
        config = AgentIPCConfig()

    if not isinstance(config, AgentIPCConfig):
        raise TypeError("config must be an AgentIPCConfig instance or None")

    if environ is None:
        environ = os.environ

    if not isinstance(environ, Mapping):
        raise TypeError("environ must be a Mapping[str, str] or None")

    # Validate all environ values are strings
    for key, value in environ.items():
        if not isinstance(key, str):
            raise TypeError("all environ keys must be str")
        if not isinstance(value, str):
            raise TypeError("all environ values must be str")

    # Resolve provider selectors
    llm_provider = _get_provider_selector(
        environ, "AGENTIPC_LLM_PROVIDER", config.llm_provider
    )
    embedding_provider = _get_provider_selector(
        environ, "AGENTIPC_EMBEDDING_PROVIDER", config.embedding_provider
    )

    # Create resolved config (no secrets)
    resolved_config = AgentIPCConfig(
        state_root=config.state_root,
        memory_root=config.memory_root,
        artifact_root=config.artifact_root,
        results_root=config.results_root,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        random_seed=config.random_seed,
    )

    # Build provider-specific options
    llm_options = _build_llm_options(llm_provider, environ)
    embedding_options = _build_embedding_options(embedding_provider, environ)

    # Construct providers via existing factory
    bundle = create_provider_bundle(
        resolved_config,
        llm_options=llm_options,
        embedding_options=embedding_options,
    )

    return ResolvedProviderBundle(config=resolved_config, bundle=bundle)


def _get_provider_selector(
    environ: Mapping[str, str],
    var_name: str,
    default: str,
) -> str:
    """Read a provider selector from environment, with validation."""
    if var_name not in environ:
        return default

    value = environ[var_name]
    if value == "":
        raise ValueError(f"{var_name} must not be empty string")

    return value


def _build_llm_options(
    provider: str,
    environ: Mapping[str, str],
) -> dict[str, object]:
    """Build LLM provider options from environment variables."""
    if provider != "openai":
        return {}

    options: dict[str, object] = {}

    # Model (passed through, factory will require it)
    if "AGENTIPC_LLM_MODEL" in environ:
        model = environ["AGENTIPC_LLM_MODEL"]
        if model == "":
            raise ValueError("AGENTIPC_LLM_MODEL must not be empty string")
        options["model"] = model

    # API key (optional for local servers)
    if "AGENTIPC_LLM_API_KEY" in environ:
        api_key = environ["AGENTIPC_LLM_API_KEY"]
        if api_key == "":
            raise ValueError("AGENTIPC_LLM_API_KEY must not be empty string")
        options["api_key"] = api_key

    # Base URL
    if "AGENTIPC_LLM_BASE_URL" in environ:
        base_url = environ["AGENTIPC_LLM_BASE_URL"]
        if base_url == "":
            raise ValueError("AGENTIPC_LLM_BASE_URL must not be empty string")
        options["base_url"] = base_url

    # Timeout
    if "AGENTIPC_LLM_TIMEOUT_SEC" in environ:
        timeout_str = environ["AGENTIPC_LLM_TIMEOUT_SEC"]
        try:
            timeout_sec = float(timeout_str)
        except ValueError as exc:
            raise ValueError(
                f"AGENTIPC_LLM_TIMEOUT_SEC must be a valid number, got {timeout_str!r}"
            ) from exc
        options["timeout_sec"] = timeout_sec

    return options


def _build_embedding_options(
    provider: str,
    environ: Mapping[str, str],
) -> dict[str, object]:
    """Build embedding provider options from environment variables."""
    if provider == "hash":
        return _build_hash_options(environ)

    if provider == "sentence-transformer":
        return _build_sentence_transformer_options(environ)

    return {}


def _build_hash_options(environ: Mapping[str, str]) -> dict[str, object]:
    """Build hash embedding options from environment."""
    options: dict[str, object] = {}

    if "AGENTIPC_EMBEDDING_DIM" in environ:
        dim_str = environ["AGENTIPC_EMBEDDING_DIM"]
        try:
            dim = int(dim_str)
        except ValueError as exc:
            raise ValueError(
                f"AGENTIPC_EMBEDDING_DIM must be a valid integer, got {dim_str!r}"
            ) from exc
        options["dim"] = dim

    return options


def _build_sentence_transformer_options(
    environ: Mapping[str, str],
) -> dict[str, object]:
    """Build sentence-transformer options from environment."""
    options: dict[str, object] = {}

    # Model name (passed through, factory will require it)
    if "AGENTIPC_EMBEDDING_MODEL_NAME" in environ:
        model_name = environ["AGENTIPC_EMBEDDING_MODEL_NAME"]
        if model_name == "":
            raise ValueError("AGENTIPC_EMBEDDING_MODEL_NAME must not be empty string")
        options["model_name"] = model_name

    # Device
    if "AGENTIPC_EMBEDDING_DEVICE" in environ:
        device = environ["AGENTIPC_EMBEDDING_DEVICE"]
        if device == "":
            raise ValueError("AGENTIPC_EMBEDDING_DEVICE must not be empty string")
        options["device"] = device

    # Local files only (strict bool parsing)
    if "AGENTIPC_EMBEDDING_LOCAL_FILES_ONLY" in environ:
        local_files_only_str = environ["AGENTIPC_EMBEDDING_LOCAL_FILES_ONLY"]
        local_files_only = _parse_bool(local_files_only_str)
        options["local_files_only"] = local_files_only

    return options


def _parse_bool(value: str) -> bool:
    """Parse a boolean from string, accepting only true/false case-insensitively."""
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False

    raise ValueError(
        f"boolean environment variable must be 'true' or 'false' "
        f"(case-insensitive), got {value!r}"
    )
