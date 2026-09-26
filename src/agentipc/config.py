from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class AgentIPCConfig(BaseModel):
    """Deterministic, offline-safe configuration for AgentIPC."""

    model_config = ConfigDict(extra="forbid")

    state_root: Path = Path(".agentipc/state")
    memory_root: Path = Path(".agentipc/memory")
    artifact_root: Path = Path(".agentipc/artifacts")
    results_root: Path = Path("results")

    llm_provider: str = Field(default="mock", min_length=1)
    embedding_provider: str = Field(default="hash", min_length=1)
    random_seed: int = Field(default=42, ge=0)


def load_config(path: str | Path) -> AgentIPCConfig:
    """Load an AgentIPC configuration from a YAML file.

    The loader is intentionally explicit: it reads only the provided file and
    does not inspect .env files or business environment variables.
    """

    with Path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        data = {}

    return AgentIPCConfig.model_validate(data)