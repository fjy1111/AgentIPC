from __future__ import annotations

from dataclasses import dataclass
from multiprocessing.shared_memory import SharedMemory
from urllib.parse import urlsplit

import numpy as np

from agentipc.protocol.refs import StateRef
from agentipc.state.checksum import array_checksum
from agentipc.state.inproc import InProcessArrayStore
from agentipc.state.shared_memory import (
    SharedArrayMetadata,
    read_shared_array,
    release_shared_array,
    write_shared_array,
)


_VALID_PUT_TRANSPORTS = ("auto", "shm", "inproc")
_VALID_REF_TRANSPORTS = ("shm", "inproc")


@dataclass(frozen=True, slots=True)
class _OwnedState:
    transport: str
    resource_id: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    checksum: str
    shm_metadata: SharedArrayMetadata | None = None


class StateHub:
    def __init__(
        self,
        *,
        transport: str = "auto",
    ) -> None:
        if transport not in _VALID_PUT_TRANSPORTS:
            raise ValueError(
                "transport must be one of: auto, shm, inproc"
            )

        self._transport = transport
        self._inproc = InProcessArrayStore()
        self._owned: dict[tuple[str, str], _OwnedState] = {}
        self._closed = False

    def put_array(
        self,
        array: np.ndarray,
        *,
        kind: str,
        summary: str,
    ) -> StateRef:
        self._ensure_open()
        self._validate_put_input(array, kind=kind, summary=summary)
        checksum = array_checksum(array)

        if self._transport == "inproc":
            return self._put_inproc(
                array,
                kind=kind,
                summary=summary,
                checksum=checksum,
            )

        if self._transport == "shm":
            return self._put_shm(
                array,
                kind=kind,
                summary=summary,
                checksum=checksum,
            )

        try:
            metadata = write_shared_array(array)
        except OSError:
            return self._put_inproc(
                array,
                kind=kind,
                summary=summary,
                checksum=checksum,
            )
        return self._store_owned_shm(
            array,
            metadata=metadata,
            kind=kind,
            summary=summary,
            checksum=checksum,
        )

    def resolve_array(
        self,
        ref: StateRef,
    ) -> np.ndarray:
        self._ensure_open()
        transport, resource_id = self._parse_ref(ref)
        owned = self._owned.get((transport, resource_id))

        if owned is not None:
            self._verify_owned_ref_metadata(ref, owned)

        if transport == "shm":
            if owned is not None:
                metadata = owned.shm_metadata
                if metadata is None:
                    raise RuntimeError(
                        "owned shm state is missing shared-memory metadata"
                    )
            else:
                shape, dtype, nbytes = self._validated_ref_metadata(ref)
                metadata = SharedArrayMetadata(
                    name=resource_id,
                    shape=shape,
                    dtype=dtype,
                    nbytes=nbytes,
                )
            array = read_shared_array(metadata)
        else:
            try:
                array = self._inproc.get(resource_id)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"inproc resource not found: {resource_id}"
                ) from exc

        self._verify_resolved_metadata(array, ref)
        actual_checksum = array_checksum(array)
        if actual_checksum != ref.checksum:
            raise ValueError(
                "checksum mismatch between resolved array and StateRef"
            )
        if owned is not None and actual_checksum != owned.checksum:
            raise ValueError(
                "checksum mismatch between resolved array and owned state"
            )

        return array

    def exists(
        self,
        ref: StateRef,
    ) -> bool:
        self._ensure_open()
        transport, resource_id = self._parse_ref(ref)

        if transport == "shm":
            try:
                shm = SharedMemory(name=resource_id)
            except FileNotFoundError:
                return False
            else:
                shm.close()
                return True

        try:
            self._inproc.get(resource_id)
        except KeyError:
            return False
        return True

    def release(
        self,
        ref: StateRef,
    ) -> None:
        transport, resource_id = self._parse_ref(ref)
        key = (transport, resource_id)
        owned = self._owned.get(key)
        if owned is None:
            return

        self._release_owned(owned)
        self._owned.pop(key, None)

    def close(self) -> None:
        if self._closed:
            return

        first_error: Exception | None = None
        for key, owned in list(self._owned.items()):
            try:
                self._release_owned(owned)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._owned.pop(key, None)

        if first_error is not None:
            raise first_error

        self._closed = True

    def __enter__(self) -> "StateHub":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _put_shm(
        self,
        array: np.ndarray,
        *,
        kind: str,
        summary: str,
        checksum: str,
    ) -> StateRef:
        metadata = write_shared_array(array)
        return self._store_owned_shm(
            array,
            metadata=metadata,
            kind=kind,
            summary=summary,
            checksum=checksum,
        )

    def _store_owned_shm(
        self,
        array: np.ndarray,
        *,
        metadata: SharedArrayMetadata,
        kind: str,
        summary: str,
        checksum: str,
    ) -> StateRef:
        try:
            ref = StateRef(
                uri=f"shm://agentipc/{metadata.name}",
                kind=kind,
                shape=list(array.shape),
                dtype=array.dtype.str,
                nbytes=array.nbytes,
                checksum=checksum,
                transport="shm",
                summary=summary,
            )
        except Exception:
            release_shared_array(metadata)
            raise

        self._owned[("shm", metadata.name)] = _OwnedState(
            transport="shm",
            resource_id=metadata.name,
            shape=tuple(array.shape),
            dtype=array.dtype.str,
            nbytes=array.nbytes,
            checksum=checksum,
            shm_metadata=metadata,
        )
        return ref

    def _put_inproc(
        self,
        array: np.ndarray,
        *,
        kind: str,
        summary: str,
        checksum: str,
    ) -> StateRef:
        key = self._inproc.put(array)
        try:
            ref = StateRef(
                uri=f"inproc://agentipc/{key}",
                kind=kind,
                shape=list(array.shape),
                dtype=array.dtype.str,
                nbytes=array.nbytes,
                checksum=checksum,
                transport="inproc",
                summary=summary,
            )
        except Exception:
            self._inproc.release(key)
            raise

        self._owned[("inproc", key)] = _OwnedState(
            transport="inproc",
            resource_id=key,
            shape=tuple(array.shape),
            dtype=array.dtype.str,
            nbytes=array.nbytes,
            checksum=checksum,
        )
        return ref

    @staticmethod
    def _validate_put_input(
        array: np.ndarray,
        *,
        kind: str,
        summary: str,
    ) -> None:
        if not isinstance(array, np.ndarray):
            raise TypeError("array must be a numpy.ndarray")
        if array.dtype.hasobject:
            raise TypeError("object dtype arrays are not supported")
        if array.nbytes == 0:
            raise ValueError("empty arrays are not supported")
        if not isinstance(kind, str):
            raise TypeError("kind must be a str")
        if not kind:
            raise ValueError("kind must be a non-empty str")
        if not isinstance(summary, str):
            raise TypeError("summary must be a str")

    @staticmethod
    def _parse_ref(ref: StateRef) -> tuple[str, str]:
        if not isinstance(ref, StateRef):
            raise TypeError("ref must be a StateRef")
        if ref.transport not in _VALID_REF_TRANSPORTS:
            raise ValueError(
                f"unknown StateRef transport: {ref.transport!r}"
            )

        parsed = urlsplit(ref.uri)
        if (
            parsed.scheme != ref.transport
            or parsed.netloc != "agentipc"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "StateRef transport and uri are inconsistent: "
                f"transport={ref.transport!r}, uri={ref.uri!r}"
            )

        resource_id = parsed.path[1:] if parsed.path.startswith("/") else ""
        if not resource_id or "/" in resource_id:
            raise ValueError(
                f"StateRef uri has invalid or empty resource identifier: {ref.uri!r}"
            )
        return ref.transport, resource_id

    @staticmethod
    def _validated_ref_metadata(
        ref: StateRef,
    ) -> tuple[tuple[int, ...], str, int]:
        if not isinstance(ref.shape, list):
            raise ValueError("StateRef shape must be a list")
        shape: list[int] = []
        for dimension in ref.shape:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
            ):
                raise ValueError("StateRef shape contains an invalid dimension")
            shape.append(dimension)

        try:
            dtype = np.dtype(ref.dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid StateRef dtype: {ref.dtype!r}") from exc

        if (
            isinstance(ref.nbytes, bool)
            or not isinstance(ref.nbytes, int)
            or ref.nbytes < 0
        ):
            raise ValueError("StateRef nbytes must be a non-negative integer")

        return tuple(shape), dtype.str, ref.nbytes

    @classmethod
    def _verify_owned_ref_metadata(
        cls,
        ref: StateRef,
        owned: _OwnedState,
    ) -> None:
        shape, dtype, nbytes = cls._validated_ref_metadata(ref)
        if shape != owned.shape:
            raise ValueError("shape mismatch for owned StateRef")
        if np.dtype(dtype) != np.dtype(owned.dtype):
            raise ValueError("dtype mismatch for owned StateRef")
        if nbytes != owned.nbytes:
            raise ValueError("nbytes mismatch for owned StateRef")

    @classmethod
    def _verify_resolved_metadata(
        cls,
        array: np.ndarray,
        ref: StateRef,
    ) -> None:
        shape, dtype, nbytes = cls._validated_ref_metadata(ref)
        if tuple(array.shape) != shape:
            raise ValueError(
                f"shape mismatch: resolved {tuple(array.shape)!r}, ref {shape!r}"
            )
        if np.dtype(array.dtype) != np.dtype(dtype):
            raise ValueError(
                f"dtype mismatch: resolved {array.dtype!r}, ref {ref.dtype!r}"
            )
        if array.nbytes != nbytes:
            raise ValueError(
                f"nbytes mismatch: resolved {array.nbytes}, ref {nbytes}"
            )

    def _release_owned(self, owned: _OwnedState) -> None:
        if owned.transport == "shm":
            metadata = owned.shm_metadata
            if metadata is None:
                raise RuntimeError(
                    "owned shm state is missing shared-memory metadata"
                )
            release_shared_array(metadata)
            return
        if owned.transport == "inproc":
            self._inproc.release(owned.resource_id)
            return
        raise RuntimeError(
            f"owned state has unsupported transport: {owned.transport!r}"
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("StateHub is closed")