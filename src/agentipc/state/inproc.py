from uuid import uuid4

import numpy as np


class InProcessArrayStore:
    def __init__(self) -> None:
        self._arrays: dict[str, np.ndarray] = {}

    def put(self, array: np.ndarray) -> str:
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy.ndarray")
        if array.dtype.hasobject:
            raise TypeError(
                "object dtype arrays are not supported by in-process transport"
            )
        if array.nbytes == 0:
            raise ValueError(
                "empty arrays are not supported by in-process transport"
            )

        stored = np.array(array, copy=True, order="C", subok=False)

        while True:
            key = f"inproc_{uuid4().hex}"
            if key not in self._arrays:
                self._arrays[key] = stored
                return key

    def get(self, key: str) -> np.ndarray:
        return np.array(
            self._arrays[key],
            copy=True,
            order="C",
            subok=False,
        )

    def release(self, key: str) -> None:
        self._arrays.pop(key, None)