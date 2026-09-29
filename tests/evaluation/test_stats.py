"""Tests for aggregate statistics computation."""

from __future__ import annotations

import json
import math

import pytest

from agentipc.evaluation.stats import AggregateStats, aggregate_stats


class TestAggregateStatsModel:
    """Test AggregateStats data model."""

    def test_valid_stats_accepted(self):
        """Valid statistics are accepted."""
        stats = AggregateStats(
            count=10,
            mean=5.5,
            std=2.87,
            min=1.0,
            max=10.0,
        )

        assert stats.count == 10
        assert stats.mean == 5.5
        assert stats.std == 2.87
        assert stats.min == 1.0
        assert stats.max == 10.0

    def test_json_serializable(self):
        """AggregateStats can be serialized to JSON."""
        stats = AggregateStats(
            count=5,
            mean=10.0,
            std=2.5,
            min=7.0,
            max=14.0,
        )

        data = stats.model_dump(mode="json")
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = AggregateStats.model_validate(parsed)

        assert restored == stats

    def test_count_must_be_at_least_one(self):
        """count must be >= 1."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AggregateStats(
                count=0,
                mean=5.0,
                std=0.0,
                min=5.0,
                max=5.0,
            )

    def test_extra_fields_forbidden(self):
        """Extra fields are rejected."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            AggregateStats(
                count=1,
                mean=5.0,
                std=0.0,
                min=5.0,
                max=5.0,
                extra="not allowed",  # type: ignore[call-arg]
            )


class TestFixedIntegerSample:
    """Test aggregate_stats with fixed integer samples."""

    def test_fixed_integer_sample(self):
        """Fixed integer sample produces expected statistics."""
        values = [1, 2, 3, 4, 5]
        stats = aggregate_stats(values)

        assert stats.count == 5
        assert stats.mean == 3.0
        assert stats.min == 1.0
        assert stats.max == 5.0

        # Population std for [1,2,3,4,5]
        # variance = mean([(1-3)^2, (2-3)^2, (3-3)^2, (4-3)^2, (5-3)^2])
        #          = mean([4, 1, 0, 1, 4]) = 10/5 = 2.0
        # std = sqrt(2.0) ≈ 1.414
        expected_std = math.sqrt(2.0)
        assert math.isclose(stats.std, expected_std, rel_tol=1e-9)


class TestFixedFloatSample:
    """Test aggregate_stats with fixed float samples."""

    def test_fixed_float_sample(self):
        """Fixed float sample produces expected statistics."""
        values = [1.5, 2.5, 3.5, 4.5]
        stats = aggregate_stats(values)

        assert stats.count == 4
        assert stats.mean == 3.0
        assert stats.min == 1.5
        assert stats.max == 4.5

        # Population std for [1.5, 2.5, 3.5, 4.5] with mean=3.0
        # variance = mean([(1.5-3)^2, (2.5-3)^2, (3.5-3)^2, (4.5-3)^2])
        #          = mean([2.25, 0.25, 0.25, 2.25]) = 5.0/4 = 1.25
        # std = sqrt(1.25) ≈ 1.118
        expected_std = math.sqrt(1.25)
        assert math.isclose(stats.std, expected_std, rel_tol=1e-9)


class TestSingleValue:
    """Test single-value input."""

    def test_single_value_has_zero_std(self):
        """Single value produces std=0.0."""
        values = [42]
        stats = aggregate_stats(values)

        assert stats.count == 1
        assert stats.mean == 42.0
        assert stats.std == 0.0
        assert stats.min == 42.0
        assert stats.max == 42.0

    def test_single_float_value(self):
        """Single float value produces std=0.0."""
        values = [3.14]
        stats = aggregate_stats(values)

        assert stats.count == 1
        assert stats.mean == 3.14
        assert stats.std == 0.0
        assert stats.min == 3.14
        assert stats.max == 3.14


class TestNegativeValues:
    """Test that negative numeric values are accepted."""

    def test_negative_integers(self):
        """Negative integers are accepted."""
        values = [-5, -3, -1, 0, 1, 3, 5]
        stats = aggregate_stats(values)

        assert stats.count == 7
        assert stats.mean == 0.0
        assert stats.min == -5.0
        assert stats.max == 5.0

    def test_all_negative_values(self):
        """All negative values are accepted."""
        values = [-10, -20, -30]
        stats = aggregate_stats(values)

        assert stats.count == 3
        assert stats.mean == -20.0
        assert stats.min == -30.0
        assert stats.max == -10.0


class TestMixedIntFloat:
    """Test mixed int and float values."""

    def test_mixed_int_and_float(self):
        """Mixed int and float values are accepted."""
        values = [1, 2.5, 3, 4.5, 5]
        stats = aggregate_stats(values)

        assert stats.count == 5
        assert stats.mean == 3.2
        assert stats.min == 1.0
        assert stats.max == 5.0


class TestInvalidInputs:
    """Test validation of invalid inputs."""

    def test_empty_list_rejected(self):
        """Empty list is rejected."""
        with pytest.raises(ValueError, match="values must be a non-empty list"):
            aggregate_stats([])

    def test_bool_rejected(self):
        """bool values are rejected."""
        with pytest.raises(TypeError, match="values\\[0\\] must be int or float, not bool"):
            aggregate_stats([True, False])  # type: ignore[list-item]

    def test_str_rejected(self):
        """str values are rejected."""
        with pytest.raises(TypeError, match="values\\[0\\] must be int or float"):
            aggregate_stats(["1", "2", "3"])  # type: ignore[list-item]

    def test_nan_rejected(self):
        """NaN is rejected."""
        with pytest.raises(ValueError, match="values\\[0\\] must be finite"):
            aggregate_stats([float("nan")])

    def test_inf_rejected(self):
        """Infinity is rejected."""
        with pytest.raises(ValueError, match="values\\[0\\] must be finite"):
            aggregate_stats([float("inf")])

    def test_negative_inf_rejected(self):
        """Negative infinity is rejected."""
        with pytest.raises(ValueError, match="values\\[0\\] must be finite"):
            aggregate_stats([float("-inf")])

    def test_tuple_rejected(self):
        """Tuple is rejected (must be list)."""
        with pytest.raises(TypeError, match="values must be a list"):
            aggregate_stats((1, 2, 3))  # type: ignore[arg-type]

    def test_mixed_valid_and_invalid_rejected(self):
        """Mixed valid and invalid values are rejected."""
        with pytest.raises(ValueError, match="values\\[2\\] must be finite"):
            aggregate_stats([1, 2, float("nan"), 4])


class TestPopulationStandardDeviation:
    """Test that population std (not sample std) is used."""

    def test_population_std_matches_expected(self):
        """Population std matches manual calculation."""
        values = [10, 20, 30]
        stats = aggregate_stats(values)

        # Mean = 20.0
        # Population variance = mean([(10-20)^2, (20-20)^2, (30-20)^2])
        #                     = mean([100, 0, 100]) = 200/3 ≈ 66.667
        # Population std = sqrt(200/3) ≈ 8.165
        expected_std = math.sqrt(200.0 / 3.0)

        assert math.isclose(stats.std, expected_std, rel_tol=1e-9)

    def test_two_values_population_std(self):
        """Two values use population std formula."""
        values = [10, 20]
        stats = aggregate_stats(values)

        # Mean = 15.0
        # Population variance = mean([(10-15)^2, (20-15)^2])
        #                     = mean([25, 25]) = 25
        # Population std = 5.0
        expected_std = 5.0

        assert stats.std == expected_std


class TestOutputTypes:
    """Test output field types."""

    def test_count_is_int(self):
        """count field is int."""
        values = [1, 2, 3]
        stats = aggregate_stats(values)

        assert type(stats.count) is int

    def test_mean_is_float(self):
        """mean field is float."""
        values = [1, 2, 3]
        stats = aggregate_stats(values)

        assert isinstance(stats.mean, float)

    def test_std_is_float(self):
        """std field is float."""
        values = [1, 2, 3]
        stats = aggregate_stats(values)

        assert isinstance(stats.std, float)

    def test_min_is_float(self):
        """min field is float."""
        values = [1, 2, 3]
        stats = aggregate_stats(values)

        assert isinstance(stats.min, float)

    def test_max_is_float(self):
        """max field is float."""
        values = [1, 2, 3]
        stats = aggregate_stats(values)

        assert isinstance(stats.max, float)
