from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentipc.experiments.real_bailian.config import RealBailianConfig


SECRET_AUDIT_FILES = (
    "environment.json",
    "provider_usage.json",
    "raw.jsonl",
    "summary.json",
    "report.md",
)


def create_result_dir(
    results_root: str | Path,
    *,
    phase: str,
    now: datetime | None = None,
) -> Path:
    if not isinstance(phase, str) or not phase:
        raise ValueError("phase must be a non-empty str")
    timestamp = utc_now(now).strftime("%Y%m%dT%H%M%SZ")
    root = Path(results_root)
    root.mkdir(parents=True, exist_ok=True)
    result_dir = root / f"{timestamp}-real-bailian-{phase}"
    result_dir.mkdir(parents=False, exist_ok=False)
    return result_dir


def audit_result_secrets(
    result_dir: Path,
    *,
    config: RealBailianConfig,
) -> dict[str, Any]:
    findings: list[str] = []
    needles = (config.api_key, config.base_url)
    scanned = 0
    for name in SECRET_AUDIT_FILES:
        path = result_dir / name
        if not path.exists():
            findings.append(f"missing:{name}")
            continue
        scanned += 1
        content = path.read_text(encoding="utf-8")
        if any(needle and needle in content for needle in needles):
            findings.append(name)
    return {
        "passed": not findings,
        "files_scanned": scanned,
        "finding_files": findings,
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")


def utc_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if not isinstance(now, datetime):
        raise TypeError("now must be a datetime or None")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    return now.astimezone(timezone.utc)
