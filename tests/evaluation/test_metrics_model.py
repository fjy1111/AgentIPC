import math

import pytest
from pydantic import ValidationError

from agentipc.evaluation.metrics import MetricsSnapshot


COUNTERS = {
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
}
EXPECTED_KEYS = COUNTERS | {"latency_ms", "success"}


def test_defaults() -> None:
    snapshot = MetricsSnapshot()

    for metric in COUNTERS:
        assert getattr(snapshot, metric) == 0
    assert snapshot.latency_ms == 0.0
    assert snapshot.success is False


def test_exact_model_keys() -> None:
    assert set(MetricsSnapshot().model_dump()) == EXPECTED_KEYS


def test_json_round_trip() -> None:
    original = MetricsSnapshot(
        message_count=2,
        protocol_bytes=100,
        memory_used=1,
        latency_ms=12.5,
        success=True,
    )

    payload = original.model_dump_json()
    restored = MetricsSnapshot.model_validate_json(payload)

    assert restored == original


@pytest.mark.parametrize("value", [-1, True, 1.0])
def test_counter_validation_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValidationError):
        MetricsSnapshot(message_count=value)


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, -math.inf, True, "12"])
def test_latency_validation_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValidationError):
        MetricsSnapshot(latency_ms=value)


def test_latency_accepts_int_and_normalizes_to_float() -> None:
    snapshot = MetricsSnapshot(latency_ms=12)

    assert snapshot.latency_ms == 12.0
    assert type(snapshot.latency_ms) is float


@pytest.mark.parametrize("value", [1, 0, "true"])
def test_success_requires_strict_bool(value: object) -> None:
    with pytest.raises(ValidationError):
        MetricsSnapshot(success=value)


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError):
        MetricsSnapshot(unexpected=1)
