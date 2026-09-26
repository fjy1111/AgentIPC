import hashlib

import numpy as np


def array_checksum(array: np.ndarray) -> str:
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a numpy.ndarray")
    if array.dtype.hasobject:
        raise TypeError("object dtype arrays are not supported for checksums")

    contiguous = np.ascontiguousarray(array)
    payload = contiguous.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()