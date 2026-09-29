import math

import pytest

from agentipc.evaluation.metrics import MetricsCollector, MetricsSnapshot


COUNTERS = [
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
]


@pytest.mark.parametrize("metric", COUNTERS)
def test_each_counter_accumulates(metric: str) -> None:
    collector = MetricsCollector()

    collector.increment(metric)
    collector.increment(metric, 2)

    assert getattr(collector.snapshot(), metric) == 3


def test_zero_increment_is_no_op() -> None:
    collector = MetricsCollector()
    collector.increment("message_count", 0)
    assert collector.snapshot().message_count == 0


def test_negative_increment_rejected() -> None:
    with pytest.raises(ValueError):
        MetricsCollector().increment("message_count", -1)


@pytest.mark.parametrize("amount", [True, False, 1.0, "1"])
def test_non_exact_int_increment_rejected(amount: object) -> None:
    with pytest.raises(TypeError):
        MetricsCollector().increment("message_count", amount)


def test_metric_type_validation() -> None:
    with pytest.raises(TypeError):
        MetricsCollector().increment(1)


@pytest.mark.parametrize("metric", ["unknown", "latency_ms", "success"])
def test_non_incrementable_metric_rejected(metric: str) -> None:
    with pytest.raises(ValueError):
        MetricsCollector().increment(metric)


def test_set_latency_ms_normalizes_int_to_float() -> None:
    collector = MetricsCollector()
    collector.set_latency_ms(12)
    assert collector.snapshot().latency_ms == 12.0
    assert type(collector.snapshot().latency_ms) is float


@pytest.mark.parametrize("value", [True, "12", None])
def test_set_latency_ms_rejects_wrong_types(value: object) -> None:
    with pytest.raises(TypeError):
        MetricsCollector().set_latency_ms(value)


@pytest.mark.parametrize("value", [-1, math.nan, math.inf, -math.inf])
def test_set_latency_ms_rejects_bad_values(value: float) -> None:
    with pytest.raises(ValueError):
        MetricsCollector().set_latency_ms(value)


def test_set_success_true_and_false() -> None:
    collector = MetricsCollector()
    collector.set_success(True)
    assert collector.snapshot().success is True
    collector.set_success(False)
    assert collector.snapshot().success is False


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_set_success_rejects_non_bool(value: object) -> None:
    with pytest.raises(TypeError):
        MetricsCollector().set_success(value)


def test_snapshot_returns_metrics_snapshot() -> None:
    assert isinstance(MetricsCollector().snapshot(), MetricsSnapshot)


def test_snapshot_is_detached_from_internal_state() -> None:
    collector = MetricsCollector()
    collector.increment("message_count", 2)

    snapshot = collector.snapshot()
    snapshot.message_count = 999

    assert collector.snapshot().message_count == 2


def test_record_llm_response_with_complete_usage() -> None:
    from agentipc.providers.base import LLMResponse

    collector = MetricsCollector()
    response = LLMResponse(
        text="test response",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=1.5,
        raw=None,
    )

    collector.record_llm_response(response)
    snapshot = collector.snapshot()

    assert snapshot.llm_call_count == 1
    assert snapshot.llm_prompt_tokens == 10
    assert snapshot.llm_completion_tokens == 5
    assert snapshot.llm_total_tokens == 15
    assert snapshot.llm_usage_missing_count == 0
    assert snapshot.llm_latency_ms == 1.5


def test_record_llm_response_accumulates_multiple_calls() -> None:
    from agentipc.providers.base import LLMResponse

    collector = MetricsCollector()

    response1 = LLMResponse(
        text="first",
        prompt_tokens=11,
        completion_tokens=3,
        latency_ms=1.25,
        raw=None,
    )
    response2 = LLMResponse(
        text="second",
        prompt_tokens=20,
        completion_tokens=4,
        latency_ms=2.5,
        raw=None,
    )

    collector.record_llm_response(response1)
    collector.record_llm_response(response2)
    snapshot = collector.snapshot()

    assert snapshot.llm_call_count == 2
    assert snapshot.llm_prompt_tokens == 31
    assert snapshot.llm_completion_tokens == 7
    assert snapshot.llm_total_tokens == 38
    assert snapshot.llm_usage_missing_count == 0
    assert snapshot.llm_latency_ms == 3.75


def test_record_llm_response_with_missing_usage() -> None:
    from agentipc.providers.base import LLMResponse

    collector = MetricsCollector()
    response = LLMResponse(
        text="mock response",
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=0.0,
        raw=None,
    )

    collector.record_llm_response(response)
    snapshot = collector.snapshot()

    assert snapshot.llm_call_count == 1
    assert snapshot.llm_prompt_tokens == 0
    assert snapshot.llm_completion_tokens == 0
    assert snapshot.llm_total_tokens == 0
    assert snapshot.llm_usage_missing_count == 1
    assert snapshot.llm_latency_ms == 0.0


def test_record_llm_response_with_partial_usage_prompt_only() -> None:
    from agentipc.providers.base import LLMResponse

    collector = MetricsCollector()
    response = LLMResponse(
        text="partial",
        prompt_tokens=10,
        completion_tokens=None,
        latency_ms=1.0,
        raw=None,
    )

    collector.record_llm_response(response)
    snapshot = collector.snapshot()

    assert snapshot.llm_call_count == 1
    assert snapshot.llm_prompt_tokens == 0
    assert snapshot.llm_completion_tokens == 0
    assert snapshot.llm_total_tokens == 0
    assert snapshot.llm_usage_missing_count == 1
    assert snapshot.llm_latency_ms == 1.0


def test_record_llm_response_with_partial_usage_completion_only() -> None:
    from agentipc.providers.base import LLMResponse

    collector = MetricsCollector()
    response = LLMResponse(
        text="partial",
        prompt_tokens=None,
        completion_tokens=5,
        latency_ms=1.0,
        raw=None,
    )

    collector.record_llm_response(response)
    snapshot = collector.snapshot()

    assert snapshot.llm_call_count == 1
    assert snapshot.llm_prompt_tokens == 0
    assert snapshot.llm_completion_tokens == 0
    assert snapshot.llm_total_tokens == 0
    assert snapshot.llm_usage_missing_count == 1
    assert snapshot.llm_latency_ms == 1.0


def test_record_llm_response_rejects_wrong_type() -> None:
    collector = MetricsCollector()

    with pytest.raises(TypeError):
        collector.record_llm_response("not a response")


def test_record_llm_response_latency_accumulates_as_float() -> None:
    from agentipc.providers.base import LLMResponse

    collector = MetricsCollector()

    response1 = LLMResponse(
        text="first",
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=1.25,
        raw=None,
    )
    response2 = LLMResponse(
        text="second",
        prompt_tokens=None,
        completion_tokens=None,
        latency_ms=2,
        raw=None,
    )

    collector.record_llm_response(response1)
    collector.record_llm_response(response2)
    snapshot = collector.snapshot()

    assert snapshot.llm_latency_ms == 3.25
    assert type(snapshot.llm_latency_ms) is float
