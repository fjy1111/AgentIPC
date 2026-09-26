from multiprocessing.shared_memory import SharedMemory
from uuid import uuid4

import numpy as np
import pytest

from agentipc.state.shared_memory import (
    SharedArrayMetadata,
    read_shared_array,
    release_shared_array,
    write_shared_array,
)


def test_float32_round_trip() -> None:
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    metadata = write_shared_array(array)

    try:
        restored = read_shared_array(metadata)

        np.testing.assert_array_equal(restored, array)
        assert restored.shape == array.shape
        assert restored.dtype == array.dtype
    finally:
        release_shared_array(metadata)


def test_multidimensional_round_trip() -> None:
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    metadata = write_shared_array(array)

    try:
        restored = read_shared_array(metadata)

        np.testing.assert_array_equal(restored, array)
        assert restored.shape == (3, 4)
        assert restored.dtype == np.dtype(np.float32)
    finally:
        release_shared_array(metadata)


def test_int32_round_trip() -> None:
    array = np.array([1, 2, 3, 4], dtype=np.int32)
    metadata = write_shared_array(array)

    try:
        restored = read_shared_array(metadata)

        np.testing.assert_array_equal(restored, array)
        assert restored.dtype == np.dtype(np.int32)
    finally:
        release_shared_array(metadata)


def test_non_contiguous_input_round_trip() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    array = base[:, ::2]
    metadata = write_shared_array(array)

    try:
        restored = read_shared_array(metadata)

        np.testing.assert_array_equal(restored, array)
        assert restored.shape == array.shape
        assert restored.dtype == array.dtype
    finally:
        release_shared_array(metadata)


def test_read_returns_independent_array_copy() -> None:
    array = np.array([2.0, 4.0, 8.0], dtype=np.float32)
    metadata = write_shared_array(array)

    try:
        restored = read_shared_array(metadata)
        release_shared_array(metadata)

        np.testing.assert_array_equal(restored, array)
        assert float(restored.sum()) == pytest.approx(14.0)
    finally:
        release_shared_array(metadata)


def test_read_rejects_inconsistent_nbytes_metadata() -> None:
    array = np.arange(4, dtype=np.float32)
    metadata = write_shared_array(array)
    invalid = SharedArrayMetadata(
        name=metadata.name,
        shape=metadata.shape,
        dtype=metadata.dtype,
        nbytes=metadata.nbytes + 1,
    )

    try:
        with pytest.raises(ValueError):
            read_shared_array(invalid)
    finally:
        release_shared_array(metadata)


def test_read_raises_for_nonexistent_segment() -> None:
    metadata = SharedArrayMetadata(
        name=f"agentipc_missing_{uuid4().hex}",
        shape=(1,),
        dtype=np.dtype(np.float32).str,
        nbytes=np.dtype(np.float32).itemsize,
    )

    with pytest.raises(FileNotFoundError):
        read_shared_array(metadata)


def test_scalar_round_trip() -> None:
    array = np.array(7, dtype=np.int32)
    metadata = write_shared_array(array)

    try:
        restored = read_shared_array(metadata)

        np.testing.assert_array_equal(restored, array)
        assert restored.shape == ()
        assert restored.dtype == array.dtype
    finally:
        release_shared_array(metadata)


def test_read_rejects_segment_smaller_than_metadata() -> None:
    dtype = np.dtype(np.float32)
    shm = SharedMemory(
        create=True,
        size=dtype.itemsize,
    )

    oversized_element_count = shm.size // dtype.itemsize + 1

    metadata = SharedArrayMetadata(
        name=shm.name,
        shape=(oversized_element_count,),
        dtype=dtype.str,
        nbytes=oversized_element_count * dtype.itemsize,
    )

    try:
        with pytest.raises(ValueError):
            read_shared_array(metadata)
    finally:
        shm.close()
        shm.unlink()