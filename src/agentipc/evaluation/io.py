"""Raw benchmark result I/O for A/B/C/D experiments.

This module provides JSONL-based persistence for RawRunRecord instances,
ensuring deterministic, UTF-8-safe, append-only storage with strict validation
on both write and read operations.
"""

from __future__ import annotations

import json
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
