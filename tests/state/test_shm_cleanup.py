from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from agentipc.state.shared_memory import (
    read_shared_array,
    release_shared_array,
    write_shared_array,
)


def test_release_removes_shared_memory_segment() -> None:
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    metadata = write_shared_array(array)

    try:
        release_shared_array(metadata)

        with pytest.raises(FileNotFoundError):
            SharedMemory(name=metadata.name)
    finally:
        release_shared_array(metadata)


def test_release_prevents_future_reads() -> None:
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    metadata = write_shared_array(array)

    try:
        release_shared_array(metadata)

        with pytest.raises(FileNotFoundError):
            read_shared_array(metadata)
    finally:
        release_shared_array(metadata)


def test_duplicate_release_is_idempotent() -> None:
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    metadata = write_shared_array(array)

    try:
        release_shared_array(metadata)
        release_shared_array(metadata)
    finally:
        release_shared_array(metadata)


def test_read_copy_survives_release() -> None:
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    metadata = write_shared_array(array)

    try:
        restored = read_shared_array(metadata)
        release_shared_array(metadata)

        np.testing.assert_array_equal(restored, array)
        assert float(restored.sum()) == pytest.approx(6.0)
    finally:
        release_shared_array(metadata)