"""Tests for derived metrics computation."""

from __future__ import annotations

import json

import pytest

from agentipc.evaluation.derived import DerivedMetrics, compute_derived_metrics


class TestDerivedMetricsModel:
    """Test DerivedMetrics data model."""

    def test_valid_metrics_accepted(self):
        """Valid derived metrics are accepted."""
        metrics = DerivedMetrics(
            token_saving_rate=0.25,
            char_saving_rate=0.30,
            latency_improvement_rate=0.15,
            repeat_work_reduction_rate=0.50,
            effective_hit_rate=0.75,
        )

        assert metrics.token_saving_rate == 0.25
        assert metrics.char_saving_rate == 0.30
        assert metrics.latency_improvement_rate == 0.15
        assert metrics.repeat_work_reduction_rate == 0.50
        assert metrics.effective_hit_rate == 0.75

    def test_none_values_accepted(self):
        """None values (undefined metrics) are accepted."""
        metrics = DerivedMetrics(
            token_saving_rate=None,
            char_saving_rate=None,
            latency_improvement_rate=None,
            repeat_work_reduction_rate=None,
            effective_hit_rate=None,
        )

        assert metrics.token_saving_rate is None
        assert metrics.char_saving_rate is None

    def test_json_serializable(self):
        """DerivedMetrics can be serialized to JSON."""
        metrics = DerivedMetrics(
            token_saving_rate=0.25,
            char_saving_rate=0.30,
            latency_improvement_rate=0.15,
            repeat_work_reduction_rate=0.50,
            effective_hit_rate=0.75,
        )

        data = metrics.model_dump(mode="json")
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        restored = DerivedMetrics.model_validate(parsed)

        assert restored == metrics

    def test_none_serializes_to_null(self):
        """None values serialize to JSON null."""
        metrics = DerivedMetrics(
            token_saving_rate=None,
            char_saving_rate=0.25,
            latency_improvement_rate=None,
            repeat_work_reduction_rate=0.10,
            effective_hit_rate=None,
        )

        data = metrics.model_dump(mode="json")
        json_str = json.dumps(data)
        decoded = json.loads(json_str)

        # Semantic validation: None values become JSON null
        assert decoded["token_saving_rate"] is None
        assert decoded["latency_improvement_rate"] is None
        assert decoded["effective_hit_rate"] is None

        # Non-None values preserved
        assert decoded["char_saving_rate"] == 0.25
        assert decoded["repeat_work_reduction_rate"] == 0.10


class TestTokenSavingRate:
    """Test token_saving_rate formula."""

    def test_50_percent_token_saving(self):
        """50% token saving is computed correctly."""
        metrics = compute_derived_metrics(
            baseline_tokens=1000,
            candidate_tokens=500,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.token_saving_rate == 0.5

    def test_zero_baseline_tokens_returns_none(self):
        """Zero baseline tokens results in None (undefined)."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.token_saving_rate is None

    def test_candidate_worse_than_baseline_negative_rate(self):
        """Candidate worse than baseline produces negative rate."""
        metrics = compute_derived_metrics(
            baseline_tokens=500,
            candidate_tokens=600,  # 20% increase
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.token_saving_rate == -0.2


class TestCharSavingRate:
    """Test char_saving_rate formula."""

    def test_25_percent_char_saving(self):
        """25% char saving is computed correctly."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=1000,
            candidate_chars=750,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.char_saving_rate == 0.25

    def test_zero_baseline_chars_returns_none(self):
        """Zero baseline chars results in None (undefined)."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=100,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.char_saving_rate is None


class TestLatencyImprovementRate:
    """Test latency_improvement_rate formula."""

    def test_latency_improvement(self):
        """Latency improvement is computed correctly."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=200.0,
            candidate_latency_ms=150.0,  # 25% faster
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.latency_improvement_rate == 0.25

    def test_zero_baseline_latency_returns_none(self):
        """Zero baseline latency results in None (undefined)."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=100.0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.latency_improvement_rate is None

    def test_candidate_slower_negative_rate(self):
        """Candidate slower than baseline produces negative rate."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=100.0,
            candidate_latency_ms=120.0,  # 20% slower
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.latency_improvement_rate == -0.2


class TestRepeatWorkReductionRate:
    """Test repeat_work_reduction_rate formula."""

    def test_repeat_work_reduction(self):
        """Repeat work reduction is computed correctly."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=10,
            candidate_repeated_work=3,  # 70% reduction
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.repeat_work_reduction_rate == 0.7

    def test_zero_baseline_repeated_work_returns_none(self):
        """Zero baseline repeated work results in None (undefined)."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=5,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.repeat_work_reduction_rate is None


class TestEffectiveHitRate:
    """Test effective_hit_rate formula."""

    def test_effective_hit_rate(self):
        """Effective hit rate is computed correctly."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=10,
            memory_effective=7,
        )

        assert metrics.effective_hit_rate == 0.7

    def test_zero_memory_used_returns_none(self):
        """Zero memory_used results in None (undefined)."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.effective_hit_rate is None

    def test_all_memory_effective(self):
        """100% effective hit rate."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=0,
            baseline_chars=0,
            candidate_chars=0,
            baseline_latency_ms=0,
            candidate_latency_ms=0,
            baseline_repeated_work=0,
            candidate_repeated_work=0,
            memory_used=5,
            memory_effective=5,
        )

        assert metrics.effective_hit_rate == 1.0


class TestEqualValues:
    """Test when baseline equals candidate (0% change)."""

    def test_equal_values_produce_zero_rate(self):
        """Equal baseline and candidate produce 0.0 rate."""
        metrics = compute_derived_metrics(
            baseline_tokens=1000,
            candidate_tokens=1000,
            baseline_chars=5000,
            candidate_chars=5000,
            baseline_latency_ms=200.0,
            candidate_latency_ms=200.0,
            baseline_repeated_work=5,
            candidate_repeated_work=5,
            memory_used=10,
            memory_effective=5,
        )

        assert metrics.token_saving_rate == 0.0
        assert metrics.char_saving_rate == 0.0
        assert metrics.latency_improvement_rate == 0.0
        assert metrics.repeat_work_reduction_rate == 0.0


class TestAllZeroDenominators:
    """Test when all denominators are zero."""

    def test_all_zero_denominators_return_none(self):
        """All zero denominators produce None for all rates."""
        metrics = compute_derived_metrics(
            baseline_tokens=0,
            candidate_tokens=100,
            baseline_chars=0,
            candidate_chars=100,
            baseline_latency_ms=0,
            candidate_latency_ms=100.0,
            baseline_repeated_work=0,
            candidate_repeated_work=5,
            memory_used=0,
            memory_effective=0,
        )

        assert metrics.token_saving_rate is None
        assert metrics.char_saving_rate is None
        assert metrics.latency_improvement_rate is None
        assert metrics.repeat_work_reduction_rate is None
        assert metrics.effective_hit_rate is None


class TestInputValidation:
    """Test validation of invalid inputs."""

    def test_negative_baseline_tokens_rejected(self):
        """Negative baseline_tokens is rejected."""
        with pytest.raises(ValueError, match="baseline_tokens must be >= 0"):
            compute_derived_metrics(
                baseline_tokens=-100,
                candidate_tokens=50,
                baseline_chars=0,
                candidate_chars=0,
                baseline_latency_ms=0,
                candidate_latency_ms=0,
                baseline_repeated_work=0,
                candidate_repeated_work=0,
                memory_used=0,
                memory_effective=0,
            )

    def test_bool_rejected(self):
        """bool is rejected."""
        with pytest.raises(TypeError, match="baseline_tokens must be int or float, not bool"):
            compute_derived_metrics(
                baseline_tokens=True,  # type: ignore[arg-type]
                candidate_tokens=50,
                baseline_chars=0,
                candidate_chars=0,
                baseline_latency_ms=0,
                candidate_latency_ms=0,
                baseline_repeated_work=0,
                candidate_repeated_work=0,
                memory_used=0,
                memory_effective=0,
            )

    def test_nan_rejected(self):
        """NaN is rejected."""
        with pytest.raises(ValueError, match="baseline_tokens must be finite"):
            compute_derived_metrics(
                baseline_tokens=float("nan"),
                candidate_tokens=50,
                baseline_chars=0,
                candidate_chars=0,
                baseline_latency_ms=0,
                candidate_latency_ms=0,
                baseline_repeated_work=0,
                candidate_repeated_work=0,
                memory_used=0,
                memory_effective=0,
            )

    def test_inf_rejected(self):
        """Infinity is rejected."""
        with pytest.raises(ValueError, match="baseline_tokens must be finite"):
            compute_derived_metrics(
                baseline_tokens=float("inf"),
                candidate_tokens=50,
                baseline_chars=0,
                candidate_chars=0,
                baseline_latency_ms=0,
                candidate_latency_ms=0,
                baseline_repeated_work=0,
                candidate_repeated_work=0,
                memory_used=0,
                memory_effective=0,
            )

    def test_memory_used_must_be_exact_int(self):
        """memory_used must be exact int."""
        with pytest.raises(TypeError, match="memory_used must be exact int"):
            compute_derived_metrics(
                baseline_tokens=0,
                candidate_tokens=0,
                baseline_chars=0,
                candidate_chars=0,
                baseline_latency_ms=0,
                candidate_latency_ms=0,
                baseline_repeated_work=0,
                candidate_repeated_work=0,
                memory_used=10.5,  # type: ignore[arg-type]
                memory_effective=5,
            )

    def test_memory_effective_must_be_exact_int(self):
        """memory_effective must be exact int."""
        with pytest.raises(TypeError, match="memory_effective must be exact int"):
            compute_derived_metrics(
                baseline_tokens=0,
                candidate_tokens=0,
                baseline_chars=0,
                candidate_chars=0,
                baseline_latency_ms=0,
                candidate_latency_ms=0,
                baseline_repeated_work=0,
                candidate_repeated_work=0,
                memory_used=10,
                memory_effective=5.5,  # type: ignore[arg-type]
            )

    def test_memory_effective_greater_than_used_rejected(self):
        """memory_effective > memory_used is rejected."""
        with pytest.raises(ValueError, match="memory_effective .* must be <= memory_used"):
            compute_derived_metrics(
                baseline_tokens=0,
                candidate_tokens=0,
                baseline_chars=0,
                candidate_chars=0,
                baseline_latency_ms=0,
                candidate_latency_ms=0,
                baseline_repeated_work=0,
                candidate_repeated_work=0,
                memory_used=5,
                memory_effective=10,
            )

    def test_negative_memory_used_rejected(self):
        """Negative memory_used is rejected."""
        with pytest.raises(ValueError, match="memory_used must be >= 0"):
            compute_derived_metrics(
                baseline_tokens=0,
                candidate_tokens=0,
                baseline_chars=0,
                candidate_chars=0,
                baseline_latency_ms=0,
                candidate_latency_ms=0,
                baseline_repeated_work=0,
                candidate_repeated_work=0,
                memory_used=-5,
                memory_effective=0,
            )
