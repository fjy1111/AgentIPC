from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentipc import __version__ as agentipc_version
from agentipc.experiments.real_bailian.config import RealBailianConfig


def capture_environment_manifest(
    *,
    config: RealBailianConfig,
    repo_root: Path,
    started_at: datetime,
    repeat: int,
    task_count: int,
) -> dict[str, Any]:
    git_commit, git_dirty = _git_state(repo_root)
    public = config.public_fields()
    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "utc_started_at": started_at.astimezone(timezone.utc).isoformat(),
        "os_release": _os_release(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "llm_provider": public["llm_provider"],
        "llm_model": public["llm_model"],
        "temperature": public["temperature"],
        "embedding_provider": public["embedding_provider"],
        "embedding_model": public["embedding_model"],
        "embedding_dim": public["embedding_dim"],
        "api_region": public["api_region"],
        "state_transport": "shm",
        "repeat": repeat,
        "task_count": task_count,
        "agentipc_version": agentipc_version,
        "openai_version": _distribution_version("openai"),
        "numpy_version": _distribution_version("numpy"),
        "pydantic_version": _distribution_version("pydantic"),
        "tiktoken_version": _distribution_version("tiktoken"),
    }


def _git_state(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return commit, bool(dirty_output.strip())
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def _os_release() -> str:
    try:
        info = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        return platform.system()
    return info.get("PRETTY_NAME") or info.get("NAME") or platform.system()


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
