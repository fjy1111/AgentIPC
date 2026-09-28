from __future__ import annotations

from collections.abc import Callable
import sys
from typing import Any

try:
    import resource as _resource
except ImportError:
    _resource = None


_DEFAULT_CPU_SECONDS = 5
_DEFAULT_ADDRESS_SPACE_BYTES = 2 * 1024 * 1024 * 1024


def resource_limits_supported() -> bool:
    if not sys.platform.startswith("linux") or _resource is None:
        return False

    return all(
        hasattr(_resource, name)
        for name in (
            "RLIMIT_CPU",
            "RLIMIT_AS",
            "RLIM_INFINITY",
            "getrlimit",
            "setrlimit",
        )
    )


def build_resource_limit_preexec_fn(
    *,
    cpu_seconds: int = _DEFAULT_CPU_SECONDS,
    address_space_bytes: int = _DEFAULT_ADDRESS_SPACE_BYTES,
) -> Callable[[], None] | None:
    _validate_positive_int(cpu_seconds, field_name="cpu_seconds")
    _validate_positive_int(address_space_bytes, field_name="address_space_bytes")

    if not resource_limits_supported():
        return None

    resource_backend = _resource
    assert resource_backend is not None

    def apply_resource_limits() -> None:
        try:
            _apply_limit(
                resource_backend,
                resource_backend.RLIMIT_CPU,
                cpu_seconds,
            )
        except (OSError, ValueError):
            pass

        try:
            _apply_limit(
                resource_backend,
                resource_backend.RLIMIT_AS,
                address_space_bytes,
            )
        except (OSError, ValueError):
            pass

    return apply_resource_limits


def _validate_positive_int(value: object, *, field_name: str) -> None:
    if type(value) is not int:
        raise TypeError(f"{field_name} must be an int")
    if value <= 0:
        raise ValueError(f"{field_name} must be greater than 0")


def _apply_limit(resource_backend: Any, limit_kind: int, requested: int) -> None:
    current_soft, current_hard = resource_backend.getrlimit(limit_kind)
    infinity = resource_backend.RLIM_INFINITY

    target_soft = _tighten_limit(
        current_soft,
        requested,
        infinity=infinity,
    )
    target_hard = _tighten_limit(
        current_hard,
        requested,
        infinity=infinity,
    )
    if target_soft > target_hard:
        target_soft = target_hard

    resource_backend.setrlimit(
        limit_kind,
        (target_soft, target_hard),
    )


def _tighten_limit(current: int, requested: int, *, infinity: int) -> int:
    if current == infinity:
        return requested
    return min(current, requested)
