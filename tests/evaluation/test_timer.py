import pytest

from agentipc.evaluation.metrics import MetricsCollector, TaskTimer


def _fake_clock(monkeypatch: pytest.MonkeyPatch, values: list[int]) -> list[int]:
    calls: list[int] = []
    iterator = iter(values)

    def perf_counter_ns() -> int:
        calls.append(1)
        return next(iterator)

    monkeypatch.setattr("agentipc.evaluation.metrics.time.perf_counter_ns", perf_counter_ns)
    return calls


def test_initial_latency_is_zero() -> None:
    assert TaskTimer().latency_ms == 0.0


def test_start_stop_returns_deterministic_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_clock(monkeypatch, [1_000_000_000, 1_250_000_000])
    timer = TaskTimer()

    assert timer.start() is None
    elapsed = timer.stop()

    assert elapsed == 250.0
    assert timer.latency_ms == 250.0


def test_repeated_start_while_running_does_not_reset_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_clock(monkeypatch, [1_000_000_000, 1_250_000_000])
    timer = TaskTimer()

    timer.start()
    timer.start()
    elapsed = timer.stop()

    assert elapsed == 250.0
    assert len(calls) == 2


def test_repeated_stop_is_idempotent_and_does_not_reread_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_clock(monkeypatch, [1_000_000_000, 1_250_000_000])
    timer = TaskTimer()
    timer.start()

    first = timer.stop()
    second = timer.stop()

    assert first == second == 250.0
    assert len(calls) == 2


def test_stop_before_start_raises() -> None:
    with pytest.raises(RuntimeError, match="not been started"):
        TaskTimer().stop()


def test_start_after_completed_stop_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_clock(monkeypatch, [1_000_000_000, 1_250_000_000])
    timer = TaskTimer()
    timer.start()
    timer.stop()

    with pytest.raises(RuntimeError, match="cannot be restarted"):
        timer.start()


def test_backward_fake_clock_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_clock(monkeypatch, [1_250_000_000, 1_000_000_000])
    timer = TaskTimer()
    timer.start()

    with pytest.raises(RuntimeError, match="backwards"):
        timer.stop()


def test_timer_and_collector_compose(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_clock(monkeypatch, [1_000_000_000, 1_250_000_000])
    timer = TaskTimer()
    collector = MetricsCollector()

    timer.start()
    elapsed = timer.stop()
    collector.set_latency_ms(elapsed)

    assert collector.snapshot().latency_ms == elapsed
