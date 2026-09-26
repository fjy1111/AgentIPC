from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from agentipc.protocol.refs import StateRef
from agentipc.state.checksum import array_checksum
from agentipc.state.hub import StateHub
from agentipc.state.shared_memory import (
    read_shared_array,
    release_shared_array,
    write_shared_array,
)


def _external_ref(array: np.ndarray):
    metadata = write_shared_array(array)
    ref = StateRef(
        uri=f"shm://agentipc/{metadata.name}",
        kind="external",
        shape=list(array.shape),
        dtype=array.dtype.str,
        nbytes=array.nbytes,
        checksum=array_checksum(array),
        transport="shm",
        summary="external",
    )
    return ref, metadata


def test_owned_shm_exists() -> None:
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="state",
            summary="",
        )
        assert hub.exists(ref) is True
    finally:
        hub.close()


def test_owned_inproc_exists() -> None:
    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="state",
            summary="",
        )
        assert hub.exists(ref) is True
    finally:
        hub.close()


@pytest.mark.parametrize("transport", ["shm", "inproc"])
def test_release_then_exists_is_false(transport: str) -> None:
    hub = StateHub(transport=transport)
    try:
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="state",
            summary="",
        )
        hub.release(ref)
        assert hub.exists(ref) is False
    finally:
        hub.close()


@pytest.mark.parametrize("transport", ["shm", "inproc"])
def test_duplicate_release_is_idempotent(transport: str) -> None:
    hub = StateHub(transport=transport)
    try:
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="state",
            summary="",
        )
        hub.release(ref)
        hub.release(ref)
    finally:
        hub.close()


@pytest.mark.parametrize("transport", ["shm", "inproc"])
def test_release_then_resolve_raises_file_not_found(transport: str) -> None:
    hub = StateHub(transport=transport)
    try:
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="state",
            summary="",
        )
        hub.release(ref)
        with pytest.raises(FileNotFoundError):
            hub.resolve_array(ref)
    finally:
        hub.close()


def test_close_cleans_remaining_owned_shm() -> None:
    hub = StateHub(transport="shm")
    ref = hub.put_array(
        np.array([1.0], dtype=np.float32),
        kind="state",
        summary="",
    )
    name = ref.uri.removeprefix("shm://agentipc/")

    hub.close()

    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def test_close_cleans_remaining_owned_inproc() -> None:
    hub = StateHub(transport="inproc")
    ref = hub.put_array(
        np.array([1.0], dtype=np.float32),
        kind="state",
        summary="",
    )
    key = ref.uri.removeprefix("inproc://agentipc/")
    store = hub._inproc

    hub.close()

    with pytest.raises(KeyError):
        store.get(key)


def test_duplicate_close_is_idempotent() -> None:
    hub = StateHub(transport="shm")
    hub.put_array(
        np.array([1.0], dtype=np.float32),
        kind="state",
        summary="",
    )
    hub.close()
    hub.close()


def test_context_manager_normal_exit_cleans_resources() -> None:
    with StateHub(transport="shm") as hub:
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="state",
            summary="",
        )
        name = ref.uri.removeprefix("shm://agentipc/")
        assert hub.exists(ref)

    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def test_context_manager_exception_exit_cleans_and_does_not_suppress() -> None:
    name = ""
    with pytest.raises(RuntimeError, match="boom"):
        with StateHub(transport="shm") as hub:
            ref = hub.put_array(
                np.array([1.0], dtype=np.float32),
                kind="state",
                summary="",
            )
            name = ref.uri.removeprefix("shm://agentipc/")
            raise RuntimeError("boom")

    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)


def test_external_shm_release_is_noop_and_backend_remains_readable() -> None:
    array = np.array([2.0, 3.0], dtype=np.float32)
    ref, metadata = _external_ref(array)
    hub = StateHub()
    try:
        hub.release(ref)
        np.testing.assert_array_equal(read_shared_array(metadata), array)
    finally:
        hub.close()
        release_shared_array(metadata)


def test_close_does_not_release_external_shm() -> None:
    array = np.array([5.0, 6.0], dtype=np.float32)
    ref, metadata = _external_ref(array)
    hub = StateHub(transport="inproc")
    try:
        assert hub.exists(ref)
        hub.close()
        np.testing.assert_array_equal(read_shared_array(metadata), array)
    finally:
        hub.close()
        release_shared_array(metadata)


def test_successful_close_blocks_put_array() -> None:
    hub = StateHub()
    hub.close()
    with pytest.raises(RuntimeError, match="closed"):
        hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="state",
            summary="",
        )


def test_successful_close_blocks_resolve_array() -> None:
    source = StateHub(transport="inproc")
    ref = source.put_array(
        np.array([1.0], dtype=np.float32),
        kind="state",
        summary="",
    )
    hub = StateHub()
    hub.close()
    try:
        with pytest.raises(RuntimeError, match="closed"):
            hub.resolve_array(ref)
    finally:
        source.close()


def test_successful_close_blocks_exists() -> None:
    ref, metadata = _external_ref(np.array([1.0], dtype=np.float32))
    hub = StateHub()
    hub.close()
    try:
        with pytest.raises(RuntimeError, match="closed"):
            hub.exists(ref)
    finally:
        release_shared_array(metadata)


def test_successful_close_blocks_enter() -> None:
    hub = StateHub()
    hub.close()
    with pytest.raises(RuntimeError, match="closed"):
        hub.__enter__()


def test_release_old_ref_after_close_is_noop() -> None:
    hub = StateHub(transport="shm")
    ref = hub.put_array(
        np.array([1.0], dtype=np.float32),
        kind="state",
        summary="",
    )
    hub.close()
    hub.release(ref)


def test_release_uses_owned_snapshot_not_tampered_shape_or_dtype() -> None:
    hub = StateHub(transport="shm")
    ref = hub.put_array(
        np.arange(4, dtype=np.float32),
        kind="state",
        summary="",
    )
    name = ref.uri.removeprefix("shm://agentipc/")
    tampered = ref.model_copy(
        update={"shape": [1], "dtype": np.dtype(np.uint8).str, "nbytes": 1}
    )

    hub.release(tampered)

    with pytest.raises(FileNotFoundError):
        SharedMemory(name=name)
    hub.close()


def test_exists_rejects_transport_uri_mismatch() -> None:
    hub = StateHub()
    ref = StateRef(
        uri="inproc://agentipc/key",
        kind="state",
        shape=[1],
        dtype=np.dtype(np.float32).str,
        nbytes=4,
        checksum="a" * 64,
        transport="shm",
        summary="",
    )
    try:
        with pytest.raises(ValueError, match="transport.*uri|uri.*transport"):
            hub.exists(ref)
    finally:
        hub.close()


def test_exists_rejects_non_state_ref() -> None:
    hub = StateHub()
    try:
        with pytest.raises(TypeError):
            hub.exists("bad")  # type: ignore[arg-type]
    finally:
        hub.close()