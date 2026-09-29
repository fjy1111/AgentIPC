"""Raw benchmark result I/O for A/B/C/D experiments.

This module provides JSONL-based persistence for RawRunRecord instances,
ensuring deterministic, UTF-8-safe, append-only storage with strict validation
on both write and read operations, plus timestamped result directory creation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from agentipc.evaluation.runner import RawRunRecord


def append_raw_record(
    path: str | Path,
    record: RawRunRecord,
) -> None:
    """Append one RawRunRecord to a JSONL file.

    This function serializes the record to compact, deterministic JSON and
    appends it as a single line to the target file. Parent directories are
    created automatically if they do not exist.

    Args:
        path: File path (str or Path) where the record will be appended
        record: RawRunRecord instance to serialize and write

    Raises:
        TypeError: If path or record have wrong types
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or Path")
    if not isinstance(record, RawRunRecord):
        raise TypeError("record must be a RawRunRecord")

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize to JSON-compatible dict
    data = record.model_dump(mode="json")

    # Produce deterministic, compact JSON
    line = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    # Append as single JSONL line
    with resolved_path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(line)
        stream.write("\n")


def read_raw_records(
    path: str | Path,
) -> list[RawRunRecord]:
    """Read all RawRunRecord entries from a JSONL file.

    This function parses each non-empty line as a separate RawRunRecord,
    validating the schema using Pydantic. Blank lines are silently skipped,
    but invalid JSON or invalid RawRunRecord schemas propagate their errors.

    Args:
        path: File path (str or Path) to read from

    Returns:
        List of RawRunRecord instances in file order

    Raises:
        TypeError: If path has wrong type
        FileNotFoundError: If the file does not exist
        json.JSONDecodeError: If a non-empty line contains invalid JSON
        pydantic.ValidationError: If valid JSON does not match RawRunRecord schema
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or Path")

    resolved_path = Path(path)

    # Let FileNotFoundError propagate naturally
    content = resolved_path.read_text(encoding="utf-8")

    records: list[RawRunRecord] = []
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            # Skip blank lines
            continue

        # Two-stage parsing to distinguish malformed JSON from invalid schema
        # Stage 1: Parse JSON (raises json.JSONDecodeError if malformed)
        payload = json.loads(stripped)

        # Stage 2: Validate schema (raises pydantic.ValidationError if invalid)
        record = RawRunRecord.model_validate(payload)
        records.append(record)

    return records


def create_result_dir(
    results_root: str | Path,
    suite: str,
    *,
    timestamp: datetime | None = None,
) -> Path:
    """Create a timestamped result directory for one benchmark suite.

    This function creates a uniquely-named directory under results_root using
    a UTC timestamp and suite identifier. The directory name format is:

        YYYYMMDDTHHMMSSffffffZ-<suite>

    Where:
    - YYYYMMDD: Date (year, month, day)
    - HHMMSS: Time (hour, minute, second)
    - ffffff: Microseconds
    - Z: UTC marker
    - <suite>: Safe suite identifier

    The function ensures results_root exists (creating it if needed), validates
    the suite identifier for path safety, and creates the target directory with
    exist_ok=False to prevent accidental overwrites.

    Args:
        results_root: Root directory for all benchmark results (str or Path)
        suite: Suite identifier (must be a safe slug with alphanumeric, _, -, .)
        timestamp: Optional explicit UTC datetime; if None, uses current UTC time

    Returns:
        Path to the created result directory

    Raises:
        TypeError: If results_root or suite have wrong types
        ValueError: If suite is empty, contains path separators, or is unsafe;
                    if timestamp is naive (missing timezone)
        FileExistsError: If the target directory already exists
    """
    # Validate results_root type
    if not isinstance(results_root, (str, Path)):
        raise TypeError("results_root must be a str or Path")

    # Validate suite type and value
    if type(suite) is not str:
        raise TypeError("suite must be an exact str")
    if suite == "":
        raise ValueError("suite must be a non-empty str")

    # Validate suite is a safe identifier (no path traversal)
    if suite in (".", ".."):
        raise ValueError("suite must not be '.' or '..'")
    if "/" in suite or "\\" in suite:
        raise ValueError("suite must not contain path separators")

    # Validate suite starts with alphanumeric
    if not suite[0].isalnum():
        raise ValueError("suite must start with an alphanumeric character")

    # Validate suite contains only safe characters
    for char in suite:
        if not (char.isalnum() or char in ("_", "-", ".")):
            raise ValueError(
                f"suite must contain only alphanumeric, _, -, . characters; got '{char}'"
            )

    # Handle timestamp
    if timestamp is None:
        # Use current UTC time
        timestamp = datetime.now(timezone.utc)
    else:
        # Validate provided timestamp
        if not isinstance(timestamp, datetime):
            raise TypeError("timestamp must be a datetime or None")
        if timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware, not naive")

        # Convert to UTC
        timestamp = timestamp.astimezone(timezone.utc)

    # Format timestamp as YYYYMMDDTHHMMSSffffffZ
    timestamp_str = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    # Build directory name
    dir_name = f"{timestamp_str}-{suite}"

    # Resolve paths
    root = Path(results_root)
    target = root / dir_name

    # Ensure root exists
    root.mkdir(parents=True, exist_ok=True)

    # Create target directory (must not exist)
    target.mkdir(exist_ok=False)

    return target
