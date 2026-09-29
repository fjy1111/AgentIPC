"""Environment snapshot capture for benchmark reproducibility.

This module captures the runtime environment at benchmark execution time,
including OS, Python version, platform details, provider configuration,
token counting method, and installed dependency versions.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform

from pydantic import BaseModel, ConfigDict, Field

from agentipc.config import AgentIPCConfig
from agentipc.evaluation.text_counter import TextCounter


class EnvironmentSnapshot(BaseModel):
    """Reproducibility metadata for one benchmark run.

    Captures the execution environment so results can be linked to their
    runtime context. This model must be JSON-serializable and must NOT
    contain secrets (API keys, tokens, base URLs, or environment dumps).
    """

    model_config = ConfigDict(extra="forbid")

    os_name: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    python_implementation: str = Field(min_length=1)

    llm_provider: str = Field(min_length=1)
    embedding_provider: str = Field(min_length=1)

    token_method: str = Field(min_length=1)

    dependencies: dict[str, str | None]


def _distribution_version(name: str) -> str | None:
    """Get installed version of a distribution, or None if not found.

    Args:
        name: Distribution name (e.g., "pydantic", "numpy")

    Returns:
        Version string if installed, None otherwise
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def capture_environment_snapshot(
    config: AgentIPCConfig,
) -> EnvironmentSnapshot:
    """Capture the current runtime environment for benchmark reproducibility.

    This function reads OS, Python, and platform metadata from standard library
    calls, extracts provider configuration from the config object, determines
    the token counting method, and checks installed dependency versions.

    Security: This function does NOT capture API keys, base URLs, access tokens,
    or the full os.environ. Provider mode (mock/openai/hash) is recorded, but
    credentials are never included.

    Args:
        config: The AgentIPCConfig instance from which to extract provider settings

    Returns:
        EnvironmentSnapshot with all environment metadata

    Raises:
        TypeError: If config is not an AgentIPCConfig instance
    """
    if not isinstance(config, AgentIPCConfig):
        raise TypeError("config must be an AgentIPCConfig")

    # Capture OS and platform metadata
    os_name = os.name
    platform_str = platform.platform()
    python_version = platform.python_version()
    python_implementation = platform.python_implementation()

    # Extract provider configuration (mode only, no credentials)
    llm_provider = config.llm_provider
    embedding_provider = config.embedding_provider

    # Determine token counting method
    # Use TextCounter to detect whether tiktoken is available
    counter = TextCounter()
    token_method = counter.count("").token_method

    # Capture dependency versions for core and optional packages
    dependencies = {
        "pydantic": _distribution_version("pydantic"),
        "numpy": _distribution_version("numpy"),
        "psutil": _distribution_version("psutil"),
        "PyYAML": _distribution_version("PyYAML"),
        "openai": _distribution_version("openai"),
        "tiktoken": _distribution_version("tiktoken"),
        "sentence-transformers": _distribution_version("sentence-transformers"),
    }

    return EnvironmentSnapshot(
        os_name=os_name,
        platform=platform_str,
        python_version=python_version,
        python_implementation=python_implementation,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
        token_method=token_method,
        dependencies=dependencies,
    )
