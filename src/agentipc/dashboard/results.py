"""Read benchmark result directories for the Dashboard API.

The Dashboard consumes the stable evaluation models instead of defining a
second result schema.  A repository scan is deliberately shallow: every
immediate child directory is treated as a candidate run and classified as
ready, incomplete, or invalid without allowing one bad run to abort the scan.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import BaseModel, ConfigDict

from agentipc.evaluation.env import EnvironmentSnapshot
from agentipc.evaluation.io import BenchmarkSummary

RunStatus = Literal["ready", "incomplete", "invalid"]

_REQUIRED_FILES = ("summary.json", "environment.json", "report.md")


class RunIndexEntry(BaseModel):
    """Safe metadata exposed by the run index endpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    status: RunStatus
    reason: str | None = None
    llm_provider: str | None = None
    embedding_provider: str | None = None
    total_records: int | None = None
    task_count: int | None = None
    seed_count: int | None = None
    report_title: str | None = None
    report_bytes: int | None = None


class RunDetail(BaseModel):
    """Schema-validated data for one ready benchmark run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str
    summary: BenchmarkSummary
    environment: EnvironmentSnapshot
    report_title: str
    report_bytes: int


def scan_results(results_root: str | Path) -> list[RunIndexEntry]:
    """Scan immediate result directories and return newest-first metadata.

    A missing root is a normal empty state.  A root that exists but is not a
    directory is a repository configuration error and raises ``ValueError``.
    Per-run problems are converted into ``incomplete``/``invalid`` entries so
    they cannot break the rest of the scan.
    """

    root = _coerce_root(results_root)
    if not root.exists():
        return []
    if not root.is_dir():
        raise ValueError("results_root must be a directory")

    entries: list[RunIndexEntry] = []
    for child in root.iterdir():
        if child.is_symlink():
            # Never follow a result-directory symlink.  A symlink to a
            # directory is still surfaced so operators can see why it is bad;
            # symlinks to ordinary files are ignored like ordinary files.
            try:
                points_to_directory = child.is_dir()
            except OSError:
                points_to_directory = False
            if points_to_directory:
                entries.append(_status_entry(child.name, "invalid", "result directory is symlink"))
            continue

        if not child.is_dir():
            continue

        entry, _detail = _inspect_run(child)
        entries.append(entry)

    entries.sort(key=lambda entry: entry.run_id, reverse=True)
    return entries


def load_run_detail(results_root: str | Path, run_id: str) -> RunDetail | None:
    """Return validated detail only for an existing ready direct child run.

    The lookup never joins the user-provided ``run_id`` onto the repository
    root.  Instead it validates the identifier and matches it against actual
    direct children, preventing traversal outside ``results_root``.
    """

    root = _coerce_root(results_root)
    if not _is_safe_run_id(run_id):
        return None
    if not root.exists():
        return None
    if not root.is_dir():
        raise ValueError("results_root must be a directory")

    for child in root.iterdir():
        if child.name != run_id:
            continue
        if child.is_symlink() or not child.is_dir():
            return None
        entry, detail = _inspect_run(child)
        if entry.status != "ready":
            return None
        return detail

    return None


def _coerce_root(results_root: str | Path) -> Path:
    if not isinstance(results_root, (str, Path)):
        raise TypeError("results_root must be a str or Path")
    return Path(results_root)


def _is_safe_run_id(run_id: str) -> bool:
    if type(run_id) is not str or run_id in {"", ".", ".."}:
        return False
    if "/" in run_id or "\\" in run_id:
        return False
    if PurePosixPath(run_id).is_absolute() or PureWindowsPath(run_id).is_absolute():
        return False
    return True


def _inspect_run(run_dir: Path) -> tuple[RunIndexEntry, RunDetail | None]:
    run_id = run_dir.name

    missing = [
        name
        for name in _REQUIRED_FILES
        if not (run_dir / name).exists() and not (run_dir / name).is_symlink()
    ]
    if missing:
        reason = f"missing required files: {', '.join(missing)}"
        return _status_entry(run_id, "incomplete", reason), None

    for name in _REQUIRED_FILES:
        path = run_dir / name
        if path.is_symlink():
            return _status_entry(run_id, "invalid", f"{name} is symlink"), None
        if not path.is_file():
            return _status_entry(run_id, "invalid", f"{name} is not a regular file"), None

    summary_path = run_dir / "summary.json"
    try:
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _status_entry(run_id, "invalid", "summary.json malformed JSON"), None
    except (OSError, UnicodeError):
        return _status_entry(run_id, "invalid", "summary.json unreadable"), None

    try:
        summary = BenchmarkSummary.model_validate(summary_payload)
    except Exception:
        return _status_entry(run_id, "invalid", "summary schema invalid"), None

    environment_path = run_dir / "environment.json"
    try:
        environment_payload = json.loads(environment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return _status_entry(run_id, "invalid", "environment.json malformed JSON"), None
    except (OSError, UnicodeError):
        return _status_entry(run_id, "invalid", "environment.json unreadable"), None

    try:
        environment = EnvironmentSnapshot.model_validate(environment_payload)
    except Exception:
        return _status_entry(run_id, "invalid", "environment schema invalid"), None

    report_path = run_dir / "report.md"
    try:
        report_raw = report_path.read_bytes()
        report_text = report_raw.decode("utf-8")
    except (OSError, UnicodeError):
        return _status_entry(run_id, "invalid", "report.md unreadable"), None

    report_title = _extract_report_title(report_text)
    if report_title is None:
        return _status_entry(run_id, "invalid", "report title malformed"), None

    report_bytes = len(report_raw)
    entry = RunIndexEntry(
        run_id=run_id,
        status="ready",
        reason=None,
        llm_provider=environment.llm_provider,
        embedding_provider=environment.embedding_provider,
        total_records=summary.total_records,
        task_count=summary.task_count,
        seed_count=summary.seed_count,
        report_title=report_title,
        report_bytes=report_bytes,
    )
    detail = RunDetail(
        run_id=run_id,
        summary=summary,
        environment=environment,
        report_title=report_title,
        report_bytes=report_bytes,
    )
    return entry, detail


def _extract_report_title(report_text: str) -> str | None:
    first_non_empty = next((line.strip() for line in report_text.splitlines() if line.strip()), None)
    if first_non_empty is None or not first_non_empty.startswith("# "):
        return None
    title = first_non_empty[2:].strip()
    return title or None


def _status_entry(run_id: str, status: RunStatus, reason: str) -> RunIndexEntry:
    return RunIndexEntry(run_id=run_id, status=status, reason=reason)
