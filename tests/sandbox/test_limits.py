from __future__ import annotations

import json
import subprocess
import sys

import pytest

from agentipc.sandbox import limits


class _FakeResource:
    RLIMIT_CPU = 1
    RLIMIT_AS = 2
    RLIM_INFINITY = -1

    def __init__(
        self,
        current: dict[int, tuple[int, int]],
        *,
        fail_on: set[int] | None = None,
    ) -> None:
        self.current = current
        self.fail_on = set() if fail_on is None else set(fail_on)
        self.set_calls: list[tuple[int, tuple[int, int]]] = []

    def getrlimit(self, kind: int) -> tuple[int, int]:
        return self.current[kind]

    def setrlimit(self, kind: int, value: tuple[int, int]) -> None:
        self.set_calls.append((kind, value))
        if kind in self.fail_on:
            raise OSError("simulated setrlimit failure")
        self.current[kind] = value


def test_default_builder_returns_callable_when_supported_or_none_otherwise() -> None:
    limiter = limits.build_resource_limit_preexec_fn()

    if limits.resource_limits_supported():
        assert callable(limiter)
    else:
        assert limiter is None


def test_explicit_limits_return_no_arg_callable_when_supported() -> None:
    limiter = limits.build_resource_limit_preexec_fn(
        cpu_seconds=2,
        address_space_bytes=512 * 1024 * 1024,
    )

    if limits.resource_limits_supported():
        assert callable(limiter)
    else:
        assert limiter is None


@pytest.mark.parametrize("value", [True, False, 1.5, "5", None])
def test_cpu_seconds_rejects_invalid_type(value: object) -> None:
    with pytest.raises(TypeError, match="cpu_seconds"):
        limits.build_resource_limit_preexec_fn(
            cpu_seconds=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0, -1])
def test_cpu_seconds_rejects_nonpositive_value(value: int) -> None:
    with pytest.raises(ValueError, match="cpu_seconds"):
        limits.build_resource_limit_preexec_fn(cpu_seconds=value)


@pytest.mark.parametrize("value", [True, False, 1.5, "1024", None])
def test_address_space_rejects_invalid_type(value: object) -> None:
    with pytest.raises(TypeError, match="address_space_bytes"):
        limits.build_resource_limit_preexec_fn(
            address_space_bytes=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0, -1])
def test_address_space_rejects_nonpositive_value(value: int) -> None:
    with pytest.raises(ValueError, match="address_space_bytes"):
        limits.build_resource_limit_preexec_fn(address_space_bytes=value)


def test_unsupported_platform_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(limits.sys, "platform", "win32")

    assert limits.resource_limits_supported() is False
    assert limits.build_resource_limit_preexec_fn() is None


def test_missing_resource_backend_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(limits, "_resource", None)

    assert limits.resource_limits_supported() is False
    assert limits.build_resource_limit_preexec_fn() is None


def test_unsupported_platform_still_validates_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(limits.sys, "platform", "win32")

    with pytest.raises(ValueError, match="cpu_seconds"):
        limits.build_resource_limit_preexec_fn(cpu_seconds=0)


def test_building_callable_does_not_mutate_parent_limits() -> None:
    if not limits.resource_limits_supported():
        pytest.skip("resource limits are not supported on this platform")

    resource_backend = limits._resource
    assert resource_backend is not None

    before_cpu = resource_backend.getrlimit(resource_backend.RLIMIT_CPU)
    before_as = resource_backend.getrlimit(resource_backend.RLIMIT_AS)

    limiter = limits.build_resource_limit_preexec_fn(
        cpu_seconds=2,
        address_space_bytes=512 * 1024 * 1024,
    )

    assert callable(limiter)
    assert resource_backend.getrlimit(resource_backend.RLIMIT_CPU) == before_cpu
    assert resource_backend.getrlimit(resource_backend.RLIMIT_AS) == before_as


def test_real_linux_child_observes_finite_requested_limits() -> None:
    if not limits.resource_limits_supported():
        pytest.skip("resource limits are not supported on this platform")

    requested_cpu = 3
    requested_as = 1024 * 1024 * 1024
    limiter = limits.build_resource_limit_preexec_fn(
        cpu_seconds=requested_cpu,
        address_space_bytes=requested_as,
    )
    assert callable(limiter)

    code = (
        "import json, resource\n"
        "print(json.dumps({"
        "'cpu': resource.getrlimit(resource.RLIMIT_CPU), "
        "'as': resource.getrlimit(resource.RLIMIT_AS)"
        "}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        preexec_fn=limiter,
        check=True,
        capture_output=True,
        text=True,
    )
    observed = json.loads(completed.stdout)

    cpu_soft, cpu_hard = observed["cpu"]
    as_soft, as_hard = observed["as"]

    assert 0 < cpu_soft <= cpu_hard <= requested_cpu
    assert 0 < as_soft <= as_hard <= requested_as


def test_existing_tighter_limits_are_not_relaxed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeResource(
        {
            _FakeResource.RLIMIT_CPU: (3, 4),
            _FakeResource.RLIMIT_AS: (100, 200),
        }
    )
    monkeypatch.setattr(limits, "_resource", fake)
    monkeypatch.setattr(limits.sys, "platform", "linux")

    limiter = limits.build_resource_limit_preexec_fn(
        cpu_seconds=5,
        address_space_bytes=300,
    )

    assert callable(limiter)
    limiter()
    assert fake.set_calls == [
        (_FakeResource.RLIMIT_CPU, (3, 4)),
        (_FakeResource.RLIMIT_AS, (100, 200)),
    ]


def test_infinite_limits_are_tightened_to_requested_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeResource(
        {
            _FakeResource.RLIMIT_CPU: (
                _FakeResource.RLIM_INFINITY,
                _FakeResource.RLIM_INFINITY,
            ),
            _FakeResource.RLIMIT_AS: (
                _FakeResource.RLIM_INFINITY,
                _FakeResource.RLIM_INFINITY,
            ),
        }
    )
    monkeypatch.setattr(limits, "_resource", fake)
    monkeypatch.setattr(limits.sys, "platform", "linux")

    limiter = limits.build_resource_limit_preexec_fn(
        cpu_seconds=5,
        address_space_bytes=300,
    )

    assert callable(limiter)
    limiter()
    assert fake.set_calls == [
        (_FakeResource.RLIMIT_CPU, (5, 5)),
        (_FakeResource.RLIMIT_AS, (300, 300)),
    ]


@pytest.mark.parametrize(
    "failed_limit",
    [
        _FakeResource.RLIMIT_CPU,
        _FakeResource.RLIMIT_AS,
    ],
)
def test_one_setrlimit_failure_does_not_skip_other_limit(
    monkeypatch: pytest.MonkeyPatch,
    failed_limit: int,
) -> None:
    fake = _FakeResource(
        {
            _FakeResource.RLIMIT_CPU: (10, 10),
            _FakeResource.RLIMIT_AS: (1000, 1000),
        },
        fail_on={failed_limit},
    )
    monkeypatch.setattr(limits, "_resource", fake)
    monkeypatch.setattr(limits.sys, "platform", "linux")

    limiter = limits.build_resource_limit_preexec_fn(
        cpu_seconds=5,
        address_space_bytes=500,
    )

    assert callable(limiter)
    limiter()
    assert [kind for kind, _ in fake.set_calls] == [
        _FakeResource.RLIMIT_CPU,
        _FakeResource.RLIMIT_AS,
    ]
