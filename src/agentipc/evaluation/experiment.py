"""Experiment configuration for A/B/C/D benchmark matrix.

This module defines the fixed experimental configurations used to evaluate
the impact of structured protocol, state exchange, and shared memory.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, StrictBool, field_validator

from agentipc.runtime.context import RunMode


class ExperimentName(str, Enum):
    """Fixed experiment identifiers for A/B/C/D matrix."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class ExperimentConfig(BaseModel):
    """Immutable experiment configuration with validated feature flags.

    Each experiment name maps to exactly one combination of mode/state/memory:
    - A: Text baseline, no state, no memory
    - B: Structured protocol, no state, no memory
    - C: Structured protocol with state exchange, no memory
    - D: Full system with structured protocol, state exchange, and shared memory

    The configuration is frozen to prevent accidental modification and uses
    strict validation to ensure name and feature flags are consistent.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ExperimentName
    mode: RunMode
    use_state: StrictBool
    use_memory: StrictBool

    @field_validator("mode", "use_state", "use_memory")
    @classmethod
    def _validate_consistency(cls, value: object, info) -> object:
        """Validate that feature flags match the experiment name.

        This validator runs after all fields are assigned, so we can check
        consistency in the root validator below.
        """
        return value

    def model_post_init(self, __context) -> None:
        """Validate that name matches feature flags after model creation."""
        expected = _EXPECTED_CONFIGS.get(self.name)
        if expected is None:
            # Should not happen since name is validated by ExperimentName enum
            raise ValueError(f"Unknown experiment name: {self.name}")

        if (
            self.mode != expected["mode"]
            or self.use_state != expected["use_state"]
            or self.use_memory != expected["use_memory"]
        ):
            raise ValueError(
                f"Experiment {self.name.value} must have mode={expected['mode'].value}, "
                f"use_state={expected['use_state']}, use_memory={expected['use_memory']}, "
                f"but got mode={self.mode.value}, use_state={self.use_state}, "
                f"use_memory={self.use_memory}"
            )


# Expected configuration for each experiment name
_EXPECTED_CONFIGS: dict[ExperimentName, dict] = {
    ExperimentName.A: {
        "mode": RunMode.TEXT,
        "use_state": False,
        "use_memory": False,
    },
    ExperimentName.B: {
        "mode": RunMode.STRUCTURED,
        "use_state": False,
        "use_memory": False,
    },
    ExperimentName.C: {
        "mode": RunMode.STRUCTURED,
        "use_state": True,
        "use_memory": False,
    },
    ExperimentName.D: {
        "mode": RunMode.STRUCTURED,
        "use_state": True,
        "use_memory": True,
    },
}


# Fixed experiment configurations
EXPERIMENT_A = ExperimentConfig(
    name=ExperimentName.A,
    mode=RunMode.TEXT,
    use_state=False,
    use_memory=False,
)

EXPERIMENT_B = ExperimentConfig(
    name=ExperimentName.B,
    mode=RunMode.STRUCTURED,
    use_state=False,
    use_memory=False,
)

EXPERIMENT_C = ExperimentConfig(
    name=ExperimentName.C,
    mode=RunMode.STRUCTURED,
    use_state=True,
    use_memory=False,
)

EXPERIMENT_D = ExperimentConfig(
    name=ExperimentName.D,
    mode=RunMode.STRUCTURED,
    use_state=True,
    use_memory=True,
)

# Fixed ordered sequence of all experiments
ABCD_CONFIGS: tuple[ExperimentConfig, ...] = (
    EXPERIMENT_A,
    EXPERIMENT_B,
    EXPERIMENT_C,
    EXPERIMENT_D,
)
