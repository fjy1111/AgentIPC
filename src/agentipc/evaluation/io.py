"""Raw benchmark result I/O for A/B/C/D experiments.

This module provides JSONL-based persistence for RawRunRecord instances,
ensuring deterministic, UTF-8-safe, append-only storage with strict validation
on both write and read operations, plus timestamped result directory creation.

It also provides summary aggregation models and writers for generating
complete A/B/C/D benchmark summaries with derived metrics.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agentipc.evaluation.derived import DerivedMetrics, compute_derived_metrics
from agentipc.evaluation.experiment import ExperimentName
from agentipc.evaluation.metrics import MetricsSnapshot
from agentipc.evaluation.runner import RawRunRecord
from agentipc.evaluation.stats import AggregateStats, aggregate_stats


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


class ExperimentSummary(BaseModel):
    """Summary statistics for one experiment configuration (A, B, C, or D).

    Aggregates all runs for one experiment across tasks and seeds, providing
    success metrics and aggregated statistics for each numeric metric field.
    """

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentName
    run_count: int = Field(ge=1, strict=True)
    success_count: int = Field(ge=0, strict=True)
    failure_count: int = Field(ge=0, strict=True)
    success_rate: float = Field(ge=0.0, le=1.0)

    metrics: dict[str, AggregateStats]

    @field_validator("success_rate")
    @classmethod
    def _validate_success_rate(cls, value: float) -> float:
        """Validate success_rate is in [0.0, 1.0]."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("success_rate must be a number")

        import math
        if not math.isfinite(value):
            raise ValueError("success_rate must be finite")

        if not (0.0 <= value <= 1.0):
            raise ValueError(f"success_rate must be in [0.0, 1.0], got {value}")

        return float(value)


class BenchmarkSummary(BaseModel):
    """Complete A/B/C/D benchmark summary with derived comparisons.

    Aggregates raw run records into experiment-level summaries and computes
    derived metrics for adjacent ablation comparisons (B vs A, C vs B, D vs C).
    """

    model_config = ConfigDict(extra="forbid")

    total_records: int = Field(ge=1, strict=True)
    task_count: int = Field(ge=1, strict=True)
    seed_count: int = Field(ge=1, strict=True)

    experiments: dict[str, ExperimentSummary]
    derived: dict[str, DerivedMetrics]


def summarize_raw_records(
    records: list[RawRunRecord],
) -> BenchmarkSummary:
    """Aggregate raw run records into a complete A/B/C/D benchmark summary.

    This function:
    1. Validates that records is a non-empty list of RawRunRecord
    2. Validates that all four experiments (A, B, C, D) are present
    3. Validates fairness: all experiments have identical (task_hash, seed) sets
    4. Validates each record's metrics against MetricsSnapshot schema
    5. Validates metrics.success matches run_result.success
    6. Aggregates numeric metrics using aggregate_stats
    7. Computes derived metrics for B vs A, C vs B, D vs C

    Args:
        records: List of RawRunRecord from A/B/C/D suite (non-empty)

    Returns:
        BenchmarkSummary with experiments and derived metrics

    Raises:
        TypeError: If records is not a list[RawRunRecord]
        ValueError: If records is empty, missing experiments, has fairness
                    violations, or contains invalid metrics
    """
    # Validate input type
    if type(records) is not list:
        raise TypeError("records must be a list")

    if len(records) == 0:
        raise ValueError("records must be a non-empty list")

    # Validate each record is RawRunRecord
    for i, record in enumerate(records):
        if not isinstance(record, RawRunRecord):
            raise TypeError(
                f"records[{i}] must be a RawRunRecord, got {type(record).__name__}"
            )

    # Group records by experiment name
    by_experiment: dict[str, list[RawRunRecord]] = {}
    for record in records:
        exp_name = record.experiment.name.value
        if exp_name not in by_experiment:
            by_experiment[exp_name] = []
        by_experiment[exp_name].append(record)

    # Validate all four experiments present
    required_experiments = {"A", "B", "C", "D"}
    actual_experiments = set(by_experiment.keys())

    if actual_experiments != required_experiments:
        missing = required_experiments - actual_experiments
        extra = actual_experiments - required_experiments
        msg_parts = []
        if missing:
            msg_parts.append(f"missing: {sorted(missing)}")
        if extra:
            msg_parts.append(f"unexpected: {sorted(extra)}")
        raise ValueError(
            f"A/B/C/D benchmark requires exactly experiments A, B, C, D; {', '.join(msg_parts)}"
        )

    # Validate fairness: all experiments have identical (task_hash, seed) sets
    experiment_keys = {}
    for exp_name in ["A", "B", "C", "D"]:
        exp_records = by_experiment[exp_name]
        keys = Counter((r.task_hash, r.seed) for r in exp_records)
        experiment_keys[exp_name] = keys

    # Check A/B/C/D have identical key sets
    a_keys = experiment_keys["A"]
    for exp_name in ["B", "C", "D"]:
        if experiment_keys[exp_name] != a_keys:
            raise ValueError(
                f"Fairness violation: experiment {exp_name} has different "
                f"(task_hash, seed) distribution than A"
            )

    # Compute task_count and seed_count from A
    a_records = by_experiment["A"]
    unique_task_hashes = {r.task_hash for r in a_records}
    unique_seeds = {r.seed for r in a_records}
    task_count = len(unique_task_hashes)
    seed_count = len(unique_seeds)

    # Aggregate each experiment
    experiment_summaries: dict[str, ExperimentSummary] = {}

    for exp_name in ["A", "B", "C", "D"]:
        exp_records = by_experiment[exp_name]

        # Validate and extract metrics from each record
        success_count = 0
        failure_count = 0

        # Collect numeric metric values for aggregation
        metric_values: dict[str, list[int | float]] = {}

        for record in exp_records:
            # Validate metrics schema
            try:
                metrics_snapshot = MetricsSnapshot.model_validate(
                    record.run_result.metrics
                )
            except Exception as e:
                raise ValueError(
                    f"Invalid metrics in experiment {exp_name}: {e}"
                ) from e

            # Validate success consistency
            if metrics_snapshot.success != record.run_result.success:
                raise ValueError(
                    f"Metrics success ({metrics_snapshot.success}) does not match "
                    f"run_result.success ({record.run_result.success}) in experiment {exp_name}"
                )

            # Count successes and failures
            if record.run_result.success:
                success_count += 1
            else:
                failure_count += 1

            # Collect numeric metrics (exclude success boolean)
            numeric_metrics = [
                "message_count",
                "text_chars",
                "text_tokens",
                "protocol_bytes",
                "state_transfer_count",
                "state_bytes",
                "artifact_ref_count",
                "memory_retrieved",
                "memory_used",
                "memory_effective",
                "memory_harmful",
                "tool_call_count",
                "repeated_tool_call_count",
                "llm_call_count",
                "llm_prompt_tokens",
                "llm_completion_tokens",
                "llm_total_tokens",
                "llm_usage_missing_count",
                "llm_latency_ms",
                "latency_ms",
            ]

            for metric_name in numeric_metrics:
                value = getattr(metrics_snapshot, metric_name)
                if metric_name not in metric_values:
                    metric_values[metric_name] = []
                metric_values[metric_name].append(value)

        # Aggregate each numeric metric
        aggregated_metrics: dict[str, AggregateStats] = {}
        for metric_name, values in metric_values.items():
            aggregated_metrics[metric_name] = aggregate_stats(values)

        # Compute success_rate
        run_count = len(exp_records)
        if success_count + failure_count != run_count:
            raise ValueError(
                f"success_count ({success_count}) + failure_count ({failure_count}) "
                f"!= run_count ({run_count}) in experiment {exp_name}"
            )

        success_rate = success_count / run_count

        # Build ExperimentSummary
        experiment_summaries[exp_name] = ExperimentSummary(
            experiment=ExperimentName(exp_name),
            run_count=run_count,
            success_count=success_count,
            failure_count=failure_count,
            success_rate=success_rate,
            metrics=aggregated_metrics,
        )

    # Compute derived metrics for adjacent ablations
    derived_metrics: dict[str, DerivedMetrics] = {}

    # B vs A: Text → Structured
    derived_metrics["B_vs_A"] = _compute_adjacent_derived(
        baseline=experiment_summaries["A"],
        candidate=experiment_summaries["B"],
    )

    # C vs B: + State
    derived_metrics["C_vs_B"] = _compute_adjacent_derived(
        baseline=experiment_summaries["B"],
        candidate=experiment_summaries["C"],
    )

    # D vs C: + Memory
    derived_metrics["D_vs_C"] = _compute_adjacent_derived(
        baseline=experiment_summaries["C"],
        candidate=experiment_summaries["D"],
    )

    # Build final summary
    return BenchmarkSummary(
        total_records=len(records),
        task_count=task_count,
        seed_count=seed_count,
        experiments=experiment_summaries,
        derived=derived_metrics,
    )


def _compute_adjacent_derived(
    *,
    baseline: ExperimentSummary,
    candidate: ExperimentSummary,
) -> DerivedMetrics:
    """Compute derived metrics for one adjacent ablation comparison.

    Args:
        baseline: The baseline experiment summary
        candidate: The candidate experiment summary

    Returns:
        DerivedMetrics with computed comparison rates
    """
    # Extract mean values for derived computation
    baseline_tokens = baseline.metrics["text_tokens"].mean
    candidate_tokens = candidate.metrics["text_tokens"].mean

    baseline_chars = baseline.metrics["text_chars"].mean
    candidate_chars = candidate.metrics["text_chars"].mean

    baseline_latency_ms = baseline.metrics["latency_ms"].mean
    candidate_latency_ms = candidate.metrics["latency_ms"].mean

    baseline_repeated_work = baseline.metrics["repeated_tool_call_count"].mean
    candidate_repeated_work = candidate.metrics["repeated_tool_call_count"].mean

    # Memory metrics use totals, not means
    # Sum across all runs in candidate experiment
    memory_used = candidate.metrics["memory_used"].count * int(
        candidate.metrics["memory_used"].mean
    )
    memory_effective = candidate.metrics["memory_effective"].count * int(
        candidate.metrics["memory_effective"].mean
    )

    return compute_derived_metrics(
        baseline_tokens=baseline_tokens,
        candidate_tokens=candidate_tokens,
        baseline_chars=baseline_chars,
        candidate_chars=candidate_chars,
        baseline_latency_ms=baseline_latency_ms,
        candidate_latency_ms=candidate_latency_ms,
        baseline_repeated_work=baseline_repeated_work,
        candidate_repeated_work=candidate_repeated_work,
        memory_used=memory_used,
        memory_effective=memory_effective,
    )


def write_summary(
    path: str | Path,
    summary: BenchmarkSummary,
) -> None:
    """Write BenchmarkSummary to a JSON file.

    This function serializes the summary to deterministic, human-readable JSON
    and writes it to the specified path. Parent directories are created
    automatically. The file is opened in exclusive create mode to prevent
    accidental overwrites.

    Args:
        path: File path (str or Path) where the summary will be written
        summary: BenchmarkSummary instance to serialize and write

    Raises:
        TypeError: If path or summary have wrong types
        FileExistsError: If the file already exists
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or Path")
    if not isinstance(summary, BenchmarkSummary):
        raise TypeError("summary must be a BenchmarkSummary")

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize to JSON-compatible dict
    data = summary.model_dump(mode="json")

    # Produce deterministic, readable JSON
    json_str = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )

    # Write with exclusive create (prevents overwrite)
    with resolved_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json_str)
        stream.write("\n")
