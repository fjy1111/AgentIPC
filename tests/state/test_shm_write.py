from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from agentipc.state.shared_memory import (
    release_shared_array,
    write_shared_array,
)


def test_write_returns_correct_metadata() -> None:
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    metadata = write_shared_array(array)

    try:
        assert metadata.name
        assert metadata.shape == (3,)
        assert np.dtype(metadata.dtype) == np.dtype(np.float32)
        assert metadata.nbytes == array.nbytes
    finally:
        release_shared_array(metadata)


def test_write_creates_reopenable_shared_memory_segment() -> None:
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    metadata = write_shared_array(array)

    try:
        shm = SharedMemory(name=metadata.name)
        try:
            assert shm.size >= metadata.nbytes
        finally:
            shm.close()
    finally:
        release_shared_array(metadata)


def test_write_preserves_multidimensional_metadata() -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    metadata = write_shared_array(array)

    try:
        assert metadata.shape == (2, 3)
        assert np.dtype(metadata.dtype) == array.dtype
        assert metadata.nbytes == array.nbytes
    finally:
        release_shared_array(metadata)


def test_write_handles_non_contiguous_input_values() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    array = base[:, ::2]
    assert not array.flags.c_contiguous

    metadata = write_shared_array(array)

    try:
        shm = SharedMemory(name=metadata.name)
        try:
            restored = np.ndarray(
                metadata.shape,
                dtype=np.dtype(metadata.dtype),
                buffer=shm.buf,
            ).copy()
        finally:
            shm.close()

        np.testing.assert_array_equal(restored, array)
    finally:
        release_shared_array(metadata)


def test_write_rejects_object_dtype() -> None:
    array = np.array([{"value": 1}], dtype=object)

    with pytest.raises(TypeError):
        write_shared_array(array)


def test_write_rejects_empty_array() -> None:
    array = np.array([], dtype=np.float32)

    with pytest.raises(ValueError):
        write_shared_array(array)


def test_write_preserves_scalar_shape() -> None:
    array = np.array(7, dtype=np.int32)
    metadata = write_shared_array(array)

    try:
        assert metadata.shape == ()
        assert np.dtype(metadata.dtype) == array.dtype
        assert metadata.nbytes == array.nbytes
    finally:
        release_shared_array(metadata)


def test_write_rejects_python_list_input() -> None:
    with pytest.raises(TypeError):
        write_shared_array([1.0, 2.0, 3.0])