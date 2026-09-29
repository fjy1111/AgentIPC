"""Tests for experiment configuration model and fixed A/B/C/D matrix."""

import pytest
from pydantic import ValidationError

from agentipc.evaluation.experiment import (
    ABCD_CONFIGS,
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    EXPERIMENT_D,
    ExperimentConfig,
    ExperimentName,
)
from agentipc.runtime.context import RunMode


class TestFixedConfigurations:
    """Test that the four fixed experiment configurations exist and are correct."""

    def test_experiment_a_exists(self):
        """EXPERIMENT_A is a valid ExperimentConfig."""
        assert isinstance(EXPERIMENT_A, ExperimentConfig)

    def test_experiment_b_exists(self):
        """EXPERIMENT_B is a valid ExperimentConfig."""
        assert isinstance(EXPERIMENT_B, ExperimentConfig)

    def test_experiment_c_exists(self):
        """EXPERIMENT_C is a valid ExperimentConfig."""
        assert isinstance(EXPERIMENT_C, ExperimentConfig)

    def test_experiment_d_exists(self):
        """EXPERIMENT_D is a valid ExperimentConfig."""
        assert isinstance(EXPERIMENT_D, ExperimentConfig)


class TestExperimentASemantics:
    """Test that A is text baseline with no infrastructure."""

    def test_name_is_a(self):
        assert EXPERIMENT_A.name == ExperimentName.A

    def test_mode_is_text(self):
        assert EXPERIMENT_A.mode is RunMode.TEXT

    def test_use_state_is_false(self):
        assert EXPERIMENT_A.use_state is False

    def test_use_memory_is_false(self):
        assert EXPERIMENT_A.use_memory is False


class TestExperimentBSemantics:
    """Test that B is structured protocol with no infrastructure."""

    def test_name_is_b(self):
        assert EXPERIMENT_B.name == ExperimentName.B

    def test_mode_is_structured(self):
        assert EXPERIMENT_B.mode is RunMode.STRUCTURED

    def test_use_state_is_false(self):
        assert EXPERIMENT_B.use_state is False

    def test_use_memory_is_false(self):
        assert EXPERIMENT_B.use_memory is False


class TestExperimentCSemantics:
    """Test that C is structured protocol with state exchange."""

    def test_name_is_c(self):
        assert EXPERIMENT_C.name == ExperimentName.C

    def test_mode_is_structured(self):
        assert EXPERIMENT_C.mode is RunMode.STRUCTURED

    def test_use_state_is_true(self):
        assert EXPERIMENT_C.use_state is True

    def test_use_memory_is_false(self):
        assert EXPERIMENT_C.use_memory is False


class TestExperimentDSemantics:
    """Test that D is full system with all infrastructure enabled."""

    def test_name_is_d(self):
        assert EXPERIMENT_D.name == ExperimentName.D

    def test_mode_is_structured(self):
        assert EXPERIMENT_D.mode is RunMode.STRUCTURED

    def test_use_state_is_true(self):
        assert EXPERIMENT_D.use_state is True

    def test_use_memory_is_true(self):
        assert EXPERIMENT_D.use_memory is True


class TestABCDConfigsOrder:
    """Test that ABCD_CONFIGS contains all four experiments in strict order."""

    def test_is_tuple(self):
        """ABCD_CONFIGS is an immutable tuple, not a list."""
        assert isinstance(ABCD_CONFIGS, tuple)

    def test_has_four_elements(self):
        assert len(ABCD_CONFIGS) == 4

    def test_strict_order(self):
        """Order must be A, B, C, D."""
        assert ABCD_CONFIGS[0] is EXPERIMENT_A
        assert ABCD_CONFIGS[1] is EXPERIMENT_B
        assert ABCD_CONFIGS[2] is EXPERIMENT_C
        assert ABCD_CONFIGS[3] is EXPERIMENT_D

    def test_order_by_name(self):
        """Verify order by checking name field."""
        names = [cfg.name for cfg in ABCD_CONFIGS]
        assert names == [
            ExperimentName.A,
            ExperimentName.B,
            ExperimentName.C,
            ExperimentName.D,
        ]


class TestInvalidMatrixRejection:
    """Test that mismatched name and feature flags are rejected."""

    def test_a_with_structured_mode_rejected(self):
        """A must use TEXT mode."""
        with pytest.raises(ValueError, match="Experiment A must have mode=text"):
            ExperimentConfig(
                name=ExperimentName.A,
                mode=RunMode.STRUCTURED,
                use_state=False,
                use_memory=False,
            )

    def test_a_with_state_enabled_rejected(self):
        """A must not use state."""
        with pytest.raises(ValueError, match="Experiment A must have.*use_state=False"):
            ExperimentConfig(
                name=ExperimentName.A,
                mode=RunMode.TEXT,
                use_state=True,
                use_memory=False,
            )

    def test_a_with_memory_enabled_rejected(self):
        """A must not use memory."""
        with pytest.raises(ValueError, match="Experiment A must have.*use_memory=False"):
            ExperimentConfig(
                name=ExperimentName.A,
                mode=RunMode.TEXT,
                use_state=False,
                use_memory=True,
            )

    def test_b_with_text_mode_rejected(self):
        """B must use STRUCTURED mode."""
        with pytest.raises(ValueError, match="Experiment B must have mode=structured"):
            ExperimentConfig(
                name=ExperimentName.B,
                mode=RunMode.TEXT,
                use_state=False,
                use_memory=False,
            )

    def test_b_with_memory_enabled_rejected(self):
        """B must not use memory."""
        with pytest.raises(ValueError, match="Experiment B must have.*use_memory=False"):
            ExperimentConfig(
                name=ExperimentName.B,
                mode=RunMode.STRUCTURED,
                use_state=False,
                use_memory=True,
            )

    def test_c_with_state_disabled_rejected(self):
        """C must use state."""
        with pytest.raises(ValueError, match="Experiment C must have.*use_state=True"):
            ExperimentConfig(
                name=ExperimentName.C,
                mode=RunMode.STRUCTURED,
                use_state=False,
                use_memory=False,
            )

    def test_c_with_memory_enabled_rejected(self):
        """C must not use memory."""
        with pytest.raises(ValueError, match="Experiment C must have.*use_memory=False"):
            ExperimentConfig(
                name=ExperimentName.C,
                mode=RunMode.STRUCTURED,
                use_state=True,
                use_memory=True,
            )

    def test_d_with_state_disabled_rejected(self):
        """D must use state."""
        with pytest.raises(ValueError, match="Experiment D must have.*use_state=True"):
            ExperimentConfig(
                name=ExperimentName.D,
                mode=RunMode.STRUCTURED,
                use_state=False,
                use_memory=True,
            )

    def test_d_with_memory_disabled_rejected(self):
        """D must use memory."""
        with pytest.raises(ValueError, match="Experiment D must have.*use_memory=True"):
            ExperimentConfig(
                name=ExperimentName.D,
                mode=RunMode.STRUCTURED,
                use_state=True,
                use_memory=False,
            )


class TestExtraFieldsRejected:
    """Test that extra fields like use_sandbox are rejected."""

    def test_extra_use_sandbox_rejected(self):
        """use_sandbox is not part of the experiment config."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ExperimentConfig(
                name=ExperimentName.A,
                mode=RunMode.TEXT,
                use_state=False,
                use_memory=False,
                use_sandbox=True,  # type: ignore[call-arg]
            )

    def test_extra_arbitrary_field_rejected(self):
        """Any extra field should be rejected."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            ExperimentConfig(
                name=ExperimentName.A,
                mode=RunMode.TEXT,
                use_state=False,
                use_memory=False,
                arbitrary_field="value",  # type: ignore[call-arg]
            )


class TestStrictBoolValidation:
    """Test that bool fields only accept actual booleans, not coercible values."""

    def test_use_state_rejects_int_one(self):
        """use_state=1 should be rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name=ExperimentName.A,
                mode=RunMode.TEXT,
                use_state=1,  # type: ignore[arg-type]
                use_memory=False,
            )

    def test_use_state_rejects_int_zero(self):
        """use_state=0 should be rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name=ExperimentName.A,
                mode=RunMode.TEXT,
                use_state=0,  # type: ignore[arg-type]
                use_memory=False,
            )

    def test_use_memory_rejects_string_false(self):
        """use_memory="false" should be rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name=ExperimentName.A,
                mode=RunMode.TEXT,
                use_state=False,
                use_memory="false",  # type: ignore[arg-type]
            )

    def test_use_memory_rejects_string_true(self):
        """use_memory="true" should be rejected."""
        with pytest.raises(ValidationError):
            ExperimentConfig(
                name=ExperimentName.D,
                mode=RunMode.STRUCTURED,
                use_state=True,
                use_memory="true",  # type: ignore[arg-type]
            )


class TestFrozenConfig:
    """Test that configs are immutable after creation."""

    def test_cannot_modify_name(self):
        """Attempting to modify name should raise an error."""
        with pytest.raises(ValidationError, match="Instance is frozen"):
            EXPERIMENT_A.name = ExperimentName.B  # type: ignore[misc]

    def test_cannot_modify_mode(self):
        """Attempting to modify mode should raise an error."""
        with pytest.raises(ValidationError, match="Instance is frozen"):
            EXPERIMENT_A.mode = RunMode.STRUCTURED  # type: ignore[misc]

    def test_cannot_modify_use_state(self):
        """Attempting to modify use_state should raise an error."""
        with pytest.raises(ValidationError, match="Instance is frozen"):
            EXPERIMENT_A.use_state = True  # type: ignore[misc]

    def test_cannot_modify_use_memory(self):
        """Attempting to modify use_memory should raise an error."""
        with pytest.raises(ValidationError, match="Instance is frozen"):
            EXPERIMENT_A.use_memory = True  # type: ignore[misc]


class TestJSONSerialization:
    """Test that configs serialize to clean JSON-compatible dicts."""

    def test_experiment_a_json_serialization(self):
        """A serializes to expected JSON structure."""
        data = EXPERIMENT_A.model_dump(mode="json")
        assert data == {
            "name": "A",
            "mode": "text",
            "use_state": False,
            "use_memory": False,
        }

    def test_experiment_d_json_serialization(self):
        """D serializes to expected JSON structure."""
        data = EXPERIMENT_D.model_dump(mode="json")
        assert data == {
            "name": "D",
            "mode": "structured",
            "use_state": True,
            "use_memory": True,
        }

    def test_all_configs_are_json_serializable(self):
        """All four configs can be serialized to JSON-compatible dicts."""
        for config in ABCD_CONFIGS:
            data = config.model_dump(mode="json")
            assert isinstance(data, dict)
            assert "name" in data
            assert "mode" in data
            assert "use_state" in data
            assert "use_memory" in data
            # Ensure values are JSON-compatible primitives
            assert isinstance(data["name"], str)
            assert isinstance(data["mode"], str)
            assert isinstance(data["use_state"], bool)
            assert isinstance(data["use_memory"], bool)
