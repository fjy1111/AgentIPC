import numpy as np
import pytest

import agentipc.state.hub as hub_module
from agentipc.state.checksum import array_checksum
from agentipc.state.hub import StateHub


def test_default_auto_prefers_shm() -> None:
    hub = StateHub()
    try:
        ref = hub.put_array(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            kind="plan_embedding",
            summary="planner state",
        )
        assert ref.transport == "shm"
        assert ref.uri.startswith("shm://agentipc/")
    finally:
        hub.close()


def test_shm_transport_returns_shm_ref() -> None:
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(
            np.array([1, 2, 3], dtype=np.int32),
            kind="numbers",
            summary="integer state",
        )
        assert ref.transport == "shm"
        assert ref.uri.startswith("shm://agentipc/")
        assert ref.uri.removeprefix("shm://agentipc/")
    finally:
        hub.close()


def test_inproc_transport_returns_inproc_ref() -> None:
    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(
            np.array([1, 2, 3], dtype=np.int32),
            kind="numbers",
            summary="integer state",
        )
        assert ref.transport == "inproc"
        assert ref.uri.startswith("inproc://agentipc/")
        assert ref.uri.removeprefix("inproc://agentipc/")
    finally:
        hub.close()


def test_state_ref_fields_match_source_array() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(
            array,
            kind="  plan state  ",
            summary="  keep spacing  ",
        )
        assert ref.kind == "  plan state  "
        assert ref.summary == "  keep spacing  "
        assert ref.shape == list(array.shape)
        assert ref.dtype == array.dtype.str
        assert ref.nbytes == array.nbytes
        assert ref.checksum == array_checksum(array)
    finally:
        hub.close()


def test_non_contiguous_array_is_supported() -> None:
    base = np.arange(20, dtype=np.float32).reshape(4, 5)
    array = base[:, ::2]
    assert not array.flags.c_contiguous

    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(array, kind="view", summary="non contiguous")
        assert ref.shape == list(array.shape)
        assert ref.dtype == array.dtype.str
        assert ref.nbytes == array.nbytes
        assert ref.checksum == array_checksum(array)
    finally:
        hub.close()


def test_scalar_array_is_supported() -> None:
    array = np.array(7, dtype=np.int32)
    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(array, kind="scalar", summary="")
        assert ref.shape == []
        assert ref.dtype == array.dtype.str
        assert ref.nbytes == array.nbytes
    finally:
        hub.close()


def test_object_dtype_is_rejected() -> None:
    hub = StateHub()
    try:
        with pytest.raises(TypeError):
            hub.put_array(
                np.array([{"value": 1}], dtype=object),
                kind="bad",
                summary="object",
            )
    finally:
        hub.close()


def test_empty_array_is_rejected() -> None:
    hub = StateHub()
    try:
        with pytest.raises(ValueError):
            hub.put_array(
                np.array([], dtype=np.float32),
                kind="empty",
                summary="empty",
            )
    finally:
        hub.close()


def test_python_list_is_rejected() -> None:
    hub = StateHub()
    try:
        with pytest.raises(TypeError):
            hub.put_array(
                [1.0, 2.0, 3.0],  # type: ignore[arg-type]
                kind="bad",
                summary="list",
            )
    finally:
        hub.close()


@pytest.mark.parametrize("transport", ["", "AUTO", "socket", "redis", None])
def test_invalid_constructor_transport_is_rejected(transport) -> None:
    with pytest.raises(ValueError, match="transport"):
        StateHub(transport=transport)


def test_auto_falls_back_to_inproc_only_for_oserror(monkeypatch) -> None:
    def fail_shm(array: np.ndarray):
        raise OSError("shared memory unavailable")

    monkeypatch.setattr(hub_module, "write_shared_array", fail_shm)
    hub = StateHub(transport="auto")
    try:
        array = np.array([3.0, 4.0], dtype=np.float32)
        ref = hub.put_array(array, kind="fallback", summary="oserror")
        assert ref.transport == "inproc"
        assert ref.uri.startswith("inproc://agentipc/")
        np.testing.assert_array_equal(hub.resolve_array(ref), array)
    finally:
        hub.close()


def test_auto_does_not_fallback_for_valueerror(monkeypatch) -> None:
    inproc_called = False

    def fail_shm(array: np.ndarray):
        raise ValueError("programming or validation error")

    original_put = hub_module.InProcessArrayStore.put

    def spy_put(self, array: np.ndarray):
        nonlocal inproc_called
        inproc_called = True
        return original_put(self, array)

    monkeypatch.setattr(hub_module, "write_shared_array", fail_shm)
    monkeypatch.setattr(hub_module.InProcessArrayStore, "put", spy_put)
    hub = StateHub(transport="auto")
    try:
        with pytest.raises(ValueError, match="validation"):
            hub.put_array(
                np.array([1.0], dtype=np.float32),
                kind="state",
                summary="",
            )
        assert not inproc_called
    finally:
        hub.close()


def test_shm_mode_does_not_fallback_for_oserror(monkeypatch) -> None:
    inproc_called = False

    def fail_shm(array: np.ndarray):
        raise OSError("shared memory unavailable")

    original_put = hub_module.InProcessArrayStore.put

    def spy_put(self, array: np.ndarray):
        nonlocal inproc_called
        inproc_called = True
        return original_put(self, array)

    monkeypatch.setattr(hub_module, "write_shared_array", fail_shm)
    monkeypatch.setattr(hub_module.InProcessArrayStore, "put", spy_put)
    hub = StateHub(transport="shm")
    try:
        with pytest.raises(OSError, match="unavailable"):
            hub.put_array(
                np.array([1.0], dtype=np.float32),
                kind="state",
                summary="",
            )
        assert not inproc_called
    finally:
        hub.close()


def test_auto_does_not_fallback_for_oserror_after_shm_write(monkeypatch) -> None:
    inproc_called = False
    real_state_ref = hub_module.StateRef

    def fail_ref_construction(*args, **kwargs):
        raise OSError("post-write construction failure")

    original_put = hub_module.InProcessArrayStore.put

    def spy_put(self, array: np.ndarray):
        nonlocal inproc_called
        inproc_called = True
        return original_put(self, array)

    monkeypatch.setattr(hub_module, "StateRef", fail_ref_construction)
    monkeypatch.setattr(hub_module.InProcessArrayStore, "put", spy_put)
    hub = StateHub(transport="auto")
    try:
        with pytest.raises(OSError, match="post-write"):
            hub.put_array(
                np.array([1.0], dtype=np.float32),
                kind="state",
                summary="",
            )
        assert not inproc_called
    finally:
        monkeypatch.setattr(hub_module, "StateRef", real_state_ref)
        hub.close()


def test_inproc_mode_never_calls_shared_memory_writer(monkeypatch) -> None:
    def unexpected_shm(array: np.ndarray):
        raise AssertionError("write_shared_array must not be called")

    monkeypatch.setattr(hub_module, "write_shared_array", unexpected_shm)
    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="state",
            summary="",
        )
        assert ref.transport == "inproc"
    finally:
        hub.close()


def test_empty_kind_is_rejected_without_stripping() -> None:
    hub = StateHub(transport="inproc")
    try:
        with pytest.raises(ValueError, match="kind"):
            hub.put_array(
                np.array([1.0], dtype=np.float32),
                kind="",
                summary="",
            )
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind=" ",
            summary="",
        )
        assert ref.kind == " "
    finally:
        hub.close()


def test_kind_and_summary_types_are_validated() -> None:
    hub = StateHub(transport="inproc")
    try:
        with pytest.raises(TypeError, match="kind"):
            hub.put_array(
                np.array([1.0], dtype=np.float32),
                kind=1,  # type: ignore[arg-type]
                summary="",
            )
        with pytest.raises(TypeError, match="summary"):
            hub.put_array(
                np.array([1.0], dtype=np.float32),
                kind="state",
                summary=1,  # type: ignore[arg-type]
            )
    finally:
        hub.close()