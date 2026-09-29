"""Aggregate statistics computation for benchmark results.

This module provides population-based statistical aggregation (mean, standard
deviation, min, max) over numeric benchmark metrics, with strict validation
and rejection of non-finite values.
"""

from __future__ import annotations

import statistics

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AggregateStats(BaseModel):
    """Population statistics for a numeric metric.

    All statistics use population formulas, not sample formulas. The standard
    deviation is computed as population standard deviation (pstdev), which
    means a single-value input produces std=0.0.

    All float fields must be finite (no NaN, inf, or -inf).
    """

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=1, strict=True)
    mean: float
    std: float
    min: float
    max: float

    @field_validator("mean", "std", "min", "max")
    @classmethod
    def _validate_finite(cls, value: float) -> float:
        """Ensure all float fields are finite."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("value must be a finite number")

        normalized = float(value)

        # Reject NaN and infinities
        import math
        if not math.isfinite(normalized):
            raise ValueError("value must be finite (not NaN or inf)")

        return normalized


def aggregate_stats(
    values: list[int | float],
) -> AggregateStats:
    """Compute population statistics over numeric values.

    This function computes count, mean, population standard deviation, min,
    and max for a list of numeric values. The input must be a non-empty list
    containing only int or float (bool is explicitly rejected). All values
    must be finite.

    The standard deviation uses the population formula (statistics.pstdev),
    which means:
    - Single value: std = 0.0
    - Multiple values: population standard deviation

    Args:
        values: Non-empty list of int or float values

    Returns:
        AggregateStats with count, mean, std, min, max

    Raises:
        TypeError: If values is not a list, or contains bool/non-numeric types
        ValueError: If values is empty, or contains NaN/inf/-inf
    """
    # Validate input type
    if type(values) is not list:
        raise TypeError("values must be a list")

    # Validate non-empty
    if len(values) == 0:
        raise ValueError("values must be a non-empty list")

    # Validate each value
    import math
    for i, value in enumerate(values):
        # Reject bool (bool is int subclass but not allowed)
        if type(value) is bool:
            raise TypeError(f"values[{i}] must be int or float, not bool")

        # Accept only int or float
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"values[{i}] must be int or float, got {type(value).__name__}"
            )

        # Reject non-finite values
        if not math.isfinite(value):
            raise ValueError(f"values[{i}] must be finite, got {value}")

    # Compute statistics
    count = len(values)
    mean = statistics.mean(values)
    std = statistics.pstdev(values)  # Population standard deviation
    min_val = min(values)
    max_val = max(values)

    return AggregateStats(
        count=count,
        mean=mean,
        std=std,
        min=min_val,
        max=max_val,
    )
