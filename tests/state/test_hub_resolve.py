from multiprocessing.shared_memory import SharedMemory
from uuid import uuid4

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


def _make_external_shm_ref(array: np.ndarray) -> tuple[StateRef, object]:
    metadata = write_shared_array(array)
    ref = StateRef(
        uri=f"shm://agentipc/{metadata.name}",
        kind="external",
        shape=list(array.shape),
        dtype=array.dtype.str,
        nbytes=array.nbytes,
        checksum=array_checksum(array),
        transport="shm",
        summary="external shared memory",
    )
    return ref, metadata


def test_shm_put_resolve_round_trip_and_downstream_calculation() -> None:
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(array, kind="vector", summary="")
        restored = hub.resolve_array(ref)
        np.testing.assert_array_equal(restored, array)
        assert float(restored @ np.ones(3, dtype=np.float32)) == pytest.approx(6.0)
    finally:
        hub.close()


def test_inproc_put_resolve_round_trip() -> None:
    array = np.array([4, 5, 6], dtype=np.int32)
    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(array, kind="vector", summary="")
        np.testing.assert_array_equal(hub.resolve_array(ref), array)
    finally:
        hub.close()


def test_multidimensional_array_round_trip() -> None:
    array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(array, kind="tensor", summary="3d")
        restored = hub.resolve_array(ref)
        np.testing.assert_array_equal(restored, array)
        assert restored.shape == (2, 3, 4)
        assert restored.dtype == array.dtype
    finally:
        hub.close()


def test_scalar_array_round_trip() -> None:
    array = np.array(11, dtype=np.int16)
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(array, kind="scalar", summary="")
        restored = hub.resolve_array(ref)
        np.testing.assert_array_equal(restored, array)
        assert restored.shape == ()
    finally:
        hub.close()


def test_non_contiguous_source_round_trip() -> None:
    base = np.arange(30, dtype=np.float32).reshape(5, 6)
    array = base[::2, 1::2]
    assert not array.flags.c_contiguous
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(array, kind="view", summary="")
        np.testing.assert_array_equal(hub.resolve_array(ref), array)
    finally:
        hub.close()


@pytest.mark.parametrize("transport", ["shm", "inproc"])
def test_resolve_returns_independent_copy(transport: str) -> None:
    array = np.array([1, 2, 3], dtype=np.int32)
    hub = StateHub(transport=transport)
    try:
        ref = hub.put_array(array, kind="vector", summary="")
        first = hub.resolve_array(ref)
        first[...] = 99
        second = hub.resolve_array(ref)
        np.testing.assert_array_equal(second, array)
    finally:
        hub.close()


def test_modified_checksum_is_rejected() -> None:
    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(
            np.array([1.0, 2.0], dtype=np.float32),
            kind="vector",
            summary="",
        )
        tampered = ref.model_copy(update={"checksum": "0" * 64})
        with pytest.raises(ValueError, match="checksum"):
            hub.resolve_array(tampered)
    finally:
        hub.close()


def test_modified_nbytes_is_rejected_for_owned_resource() -> None:
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(
            np.arange(4, dtype=np.float32),
            kind="vector",
            summary="",
        )
        tampered = ref.model_copy(update={"nbytes": ref.nbytes + 4})
        with pytest.raises(ValueError, match="nbytes"):
            hub.resolve_array(tampered)
    finally:
        hub.close()


def test_modified_shape_is_rejected_for_owned_resource() -> None:
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(
            np.arange(4, dtype=np.float32),
            kind="vector",
            summary="",
        )
        tampered = ref.model_copy(update={"shape": [2, 2]})
        with pytest.raises(ValueError, match="shape"):
            hub.resolve_array(tampered)
    finally:
        hub.close()


def test_modified_dtype_is_rejected_for_owned_resource() -> None:
    hub = StateHub(transport="shm")
    try:
        ref = hub.put_array(
            np.arange(4, dtype=np.float32),
            kind="vector",
            summary="",
        )
        tampered = ref.model_copy(update={"dtype": np.dtype(np.int32).str})
        with pytest.raises(ValueError, match="dtype"):
            hub.resolve_array(tampered)
    finally:
        hub.close()


def test_transport_uri_mismatch_is_rejected() -> None:
    hub = StateHub(transport="inproc")
    try:
        ref = hub.put_array(
            np.array([1.0], dtype=np.float32),
            kind="vector",
            summary="",
        )
        mismatched = ref.model_copy(update={"transport": "shm"})
        with pytest.raises(ValueError, match="transport.*uri|uri.*transport"):
            hub.resolve_array(mismatched)
    finally:
        hub.close()


def test_unknown_transport_is_rejected() -> None:
    ref = StateRef(
        uri="mystery://agentipc/key",
        kind="vector",
        shape=[1],
        dtype=np.dtype(np.float32).str,
        nbytes=4,
        checksum="a" * 64,
        transport="mystery",
        summary="",
    )
    hub = StateHub()
    try:
        with pytest.raises(ValueError, match="transport"):
            hub.resolve_array(ref)
    finally:
        hub.close()


def test_missing_shm_resource_raises_file_not_found() -> None:
    ref = StateRef(
        uri=f"shm://agentipc/agentipc_missing_{uuid4().hex}",
        kind="vector",
        shape=[1],
        dtype=np.dtype(np.float32).str,
        nbytes=4,
        checksum="a" * 64,
        transport="shm",
        summary="",
    )
    hub = StateHub(transport="inproc")
    try:
        with pytest.raises(FileNotFoundError):
            hub.resolve_array(ref)
    finally:
        hub.close()


def test_missing_inproc_resource_raises_file_not_found() -> None:
    ref = StateRef(
        uri=f"inproc://agentipc/inproc_{uuid4().hex}",
        kind="vector",
        shape=[1],
        dtype=np.dtype(np.float32).str,
        nbytes=4,
        checksum="a" * 64,
        transport="inproc",
        summary="",
    )
    hub = StateHub(transport="shm")
    try:
        with pytest.raises(FileNotFoundError):
            hub.resolve_array(ref)
    finally:
        hub.close()


def test_external_shm_state_ref_can_be_resolved() -> None:
    array = np.arange(8, dtype=np.float32).reshape(2, 4)
    ref, metadata = _make_external_shm_ref(array)
    hub = StateHub(transport="shm")
    try:
        np.testing.assert_array_equal(hub.resolve_array(ref), array)
    finally:
        hub.close()
        release_shared_array(metadata)


def test_constructor_transport_does_not_limit_ref_dispatch() -> None:
    array = np.array([9.0, 10.0], dtype=np.float32)
    ref, metadata = _make_external_shm_ref(array)
    hub = StateHub(transport="inproc")
    try:
        np.testing.assert_array_equal(hub.resolve_array(ref), array)
    finally:
        hub.close()
        release_shared_array(metadata)


def test_external_shm_checksum_corruption_is_rejected() -> None:
    array = np.array([1, 2, 3], dtype=np.int32)
    ref, metadata = _make_external_shm_ref(array)
    try:
        shm = SharedMemory(name=metadata.name)
        try:
            shm.buf[0] ^= 0xFF
        finally:
            shm.close()

        hub = StateHub(transport="inproc")
        try:
            with pytest.raises(ValueError, match="checksum"):
                hub.resolve_array(ref)
        finally:
            hub.close()
    finally:
        release_shared_array(metadata)


def test_external_shm_metadata_mismatch_is_rejected() -> None:
    array = np.arange(4, dtype=np.float32)
    ref, metadata = _make_external_shm_ref(array)
    try:
        bad_ref = ref.model_copy(update={"nbytes": ref.nbytes + 4})
        hub = StateHub()
        try:
            with pytest.raises(ValueError):
                hub.resolve_array(bad_ref)
        finally:
            hub.close()
    finally:
        release_shared_array(metadata)


def test_resolve_rejects_non_state_ref() -> None:
    hub = StateHub()
    try:
        with pytest.raises(TypeError):
            hub.resolve_array("not-a-ref")  # type: ignore[arg-type]
    finally:
        hub.close()


def test_invalid_or_empty_resource_identifier_is_rejected() -> None:
    hub = StateHub()
    try:
        for uri in ["shm://agentipc/", "shm://agentipc/name/extra"]:
            ref = StateRef(
                uri=uri,
                kind="vector",
                shape=[1],
                dtype=np.dtype(np.float32).str,
                nbytes=4,
                checksum="a" * 64,
                transport="shm",
                summary="",
            )
            with pytest.raises(ValueError, match="uri"):
                hub.resolve_array(ref)
    finally:
        hub.close()