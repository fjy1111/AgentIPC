"""Derived metrics computation for A/B/C/D experiment comparison.

This module provides formulas for computing relative performance metrics
(saving rates, improvement rates, hit rates) from baseline and candidate
measurements. These are pure mathematical transformations that do NOT
interpret what the underlying metrics represent.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, field_validator


class DerivedMetrics(BaseModel):
    """Derived comparison metrics between baseline and candidate.

    All rate fields are expressed as fractions (0.25 = 25%), not percentages.
    Negative rates indicate the candidate performed worse than baseline.
    None indicates the metric is undefined (zero denominator).

    This model contains only mathematical formulas. It does NOT claim that:
    - token saving equals IPC communication saving
    - protocol_bytes equals text_chars
    - any specific metric pair should be compared

    The interpretation of which metrics to compare is left to summary/report
    generation (T129/T130).
    """

    model_config = ConfigDict(extra="forbid")

    token_saving_rate: float | None
    char_saving_rate: float | None
    latency_improvement_rate: float | None
    repeat_work_reduction_rate: float | None
    effective_hit_rate: float | None


def compute_derived_metrics(
    *,
    baseline_tokens: int | float,
    candidate_tokens: int | float,
    baseline_chars: int | float,
    candidate_chars: int | float,
    baseline_latency_ms: int | float,
    candidate_latency_ms: int | float,
    baseline_repeated_work: int | float,
    candidate_repeated_work: int | float,
    memory_used: int,
    memory_effective: int,
) -> DerivedMetrics:
    """Compute derived comparison metrics from baseline and candidate measurements.

    This function applies pure mathematical formulas to compute relative changes
    between baseline and candidate configurations. All saving/improvement rates
    use the formula:

        rate = (baseline - candidate) / baseline

    Positive rates indicate candidate is better (saved/improved).
    Negative rates indicate candidate is worse (increased/regressed).
    None indicates undefined (zero denominator).

    The effective_hit_rate uses:

        rate = memory_effective / memory_used

    Args:
        baseline_tokens: Baseline token count (>= 0)
        candidate_tokens: Candidate token count (>= 0)
        baseline_chars: Baseline character count (>= 0)
        candidate_chars: Candidate character count (>= 0)
        baseline_latency_ms: Baseline latency (>= 0)
        candidate_latency_ms: Candidate latency (>= 0)
        baseline_repeated_work: Baseline repeated work count (>= 0)
        candidate_repeated_work: Candidate repeated work count (>= 0)
        memory_used: Total memory entries retrieved (exact int >= 0)
        memory_effective: Memory entries that were effective (exact int >= 0)

    Returns:
        DerivedMetrics with computed rates (float or None)

    Raises:
        TypeError: If any argument has wrong type (bool, str, etc.)
        ValueError: If any value is negative, non-finite, or memory_effective > memory_used
    """
    # Validate all numeric inputs
    _validate_numeric_input("baseline_tokens", baseline_tokens, allow_zero=True)
    _validate_numeric_input("candidate_tokens", candidate_tokens, allow_zero=True)
    _validate_numeric_input("baseline_chars", baseline_chars, allow_zero=True)
    _validate_numeric_input("candidate_chars", candidate_chars, allow_zero=True)
    _validate_numeric_input("baseline_latency_ms", baseline_latency_ms, allow_zero=True)
    _validate_numeric_input("candidate_latency_ms", candidate_latency_ms, allow_zero=True)
    _validate_numeric_input("baseline_repeated_work", baseline_repeated_work, allow_zero=True)
    _validate_numeric_input("candidate_repeated_work", candidate_repeated_work, allow_zero=True)

    # Validate memory counts (exact int)
    _validate_memory_count("memory_used", memory_used)
    _validate_memory_count("memory_effective", memory_effective)

    # Validate memory_effective <= memory_used
    if memory_effective > memory_used:
        raise ValueError(
            f"memory_effective ({memory_effective}) must be <= memory_used ({memory_used})"
        )

    # Compute token saving rate
    if baseline_tokens == 0:
        token_saving_rate = None
    else:
        token_saving_rate = (baseline_tokens - candidate_tokens) / baseline_tokens

    # Compute char saving rate
    if baseline_chars == 0:
        char_saving_rate = None
    else:
        char_saving_rate = (baseline_chars - candidate_chars) / baseline_chars

    # Compute latency improvement rate
    if baseline_latency_ms == 0:
        latency_improvement_rate = None
    else:
        latency_improvement_rate = (baseline_latency_ms - candidate_latency_ms) / baseline_latency_ms

    # Compute repeat work reduction rate
    if baseline_repeated_work == 0:
        repeat_work_reduction_rate = None
    else:
        repeat_work_reduction_rate = (
            baseline_repeated_work - candidate_repeated_work
        ) / baseline_repeated_work

    # Compute effective hit rate
    if memory_used == 0:
        effective_hit_rate = None
    else:
        effective_hit_rate = memory_effective / memory_used

    return DerivedMetrics(
        token_saving_rate=token_saving_rate,
        char_saving_rate=char_saving_rate,
        latency_improvement_rate=latency_improvement_rate,
        repeat_work_reduction_rate=repeat_work_reduction_rate,
        effective_hit_rate=effective_hit_rate,
    )


def _validate_numeric_input(
    name: str,
    value: int | float,
    *,
    allow_zero: bool = True,
) -> None:
    """Validate a numeric input parameter.

    Args:
        name: Parameter name for error messages
        value: Value to validate
        allow_zero: Whether zero is allowed

    Raises:
        TypeError: If value is bool or not int/float
        ValueError: If value is negative or non-finite
    """
    # Reject bool (bool is int subclass but not allowed)
    if type(value) is bool:
        raise TypeError(f"{name} must be int or float, not bool")

    # Accept only int or float
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be int or float, got {type(value).__name__}")

    # Check finite
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")

    # Check non-negative
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")

    # Check zero if not allowed
    if not allow_zero and value == 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _validate_memory_count(name: str, value: int) -> None:
    """Validate an exact int memory count.

    Args:
        name: Parameter name for error messages
        value: Value to validate

    Raises:
        TypeError: If value is not exact int or is bool
        ValueError: If value is negative
    """
    # Reject bool
    if type(value) is bool:
        raise TypeError(f"{name} must be exact int, not bool")

    # Must be exact int
    if type(value) is not int:
        raise TypeError(f"{name} must be exact int, got {type(value).__name__}")

    # Must be non-negative
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
