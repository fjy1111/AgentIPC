from contextlib import suppress
from dataclasses import dataclass
from math import prod
from multiprocessing.shared_memory import SharedMemory

import numpy as np


@dataclass(frozen=True, slots=True)
class SharedArrayMetadata:
    name: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


def write_shared_array(array: np.ndarray) -> SharedArrayMetadata:
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a numpy.ndarray")
    if array.dtype.hasobject:
        raise TypeError(
            "object dtype arrays are not supported by shared memory transport"
        )
    if array.nbytes == 0:
        raise ValueError(
            "empty arrays are not supported by shared memory transport"
        )

    contiguous = np.array(array, copy=True, order="C", subok=False)
    shm = SharedMemory(create=True, size=array.nbytes)

    try:
        shm.buf[: array.nbytes] = contiguous.tobytes(order="C")
        metadata = SharedArrayMetadata(
            name=shm.name,
            shape=tuple(int(dimension) for dimension in array.shape),
            dtype=array.dtype.str,
            nbytes=array.nbytes,
        )
        shm.close()
    except Exception:
        with suppress(Exception):
            shm.unlink()
        with suppress(Exception):
            shm.close()
        raise

    return metadata


def read_shared_array(
    metadata: SharedArrayMetadata,
) -> np.ndarray:
    dtype = np.dtype(metadata.dtype)
    expected_nbytes = prod(metadata.shape) * dtype.itemsize

    if expected_nbytes != metadata.nbytes:
        raise ValueError(
            "shared array metadata nbytes does not match shape and dtype: "
            f"expected {expected_nbytes}, got {metadata.nbytes}"
        )

    shm = SharedMemory(name=metadata.name)

    try:
        if shm.size < metadata.nbytes:
            raise ValueError(
                "shared memory segment is smaller than metadata nbytes: "
                f"segment has {shm.size}, metadata requires {metadata.nbytes}"
            )

        view = np.ndarray(
            metadata.shape,
            dtype=dtype,
            buffer=shm.buf,
        )
        return view.copy()
    finally:
        shm.close()


def release_shared_array(
    metadata: SharedArrayMetadata,
) -> None:
    try:
        shm = SharedMemory(name=metadata.name)
    except FileNotFoundError:
        return

    try:
        try:
            shm.unlink()
        except FileNotFoundError:
            return
    finally:
        shm.close()