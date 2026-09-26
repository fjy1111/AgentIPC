import hashlib

import numpy as np
import pytest

from agentipc.state.checksum import array_checksum


def test_checksum_is_deterministic() -> None:
    array = np.arange(8, dtype=np.float32)

    assert array_checksum(array) == array_checksum(array)


def test_equivalent_arrays_have_same_checksum() -> None:
    first = np.arange(12, dtype=np.int32).reshape(3, 4)
    second = first.copy()

    assert array_checksum(first) == array_checksum(second)


def test_content_change_changes_checksum() -> None:
    first = np.array([1, 2, 3], dtype=np.int32)
    second = first.copy()
    second[1] = 99

    assert array_checksum(first) != array_checksum(second)


def test_non_contiguous_array_matches_contiguous_copy() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    view = base[:, ::2]
    contiguous = np.ascontiguousarray(view)
    assert not view.flags.c_contiguous

    assert array_checksum(view) == array_checksum(contiguous)


def test_checksum_matches_known_sha256_calculation() -> None:
    array = np.array([1, 2, 3, 4], dtype=np.uint16)
    expected = hashlib.sha256(
        np.ascontiguousarray(array).tobytes(order="C")
    ).hexdigest()

    assert array_checksum(array) == expected


def test_checksum_format_is_lowercase_sha256_hex() -> None:
    checksum = array_checksum(np.array([1.0], dtype=np.float32))

    assert len(checksum) == 64
    assert set(checksum) <= set("0123456789abcdef")


def test_checksum_rejects_python_list() -> None:
    with pytest.raises(TypeError):
        array_checksum([1.0, 2.0, 3.0])


def test_checksum_rejects_object_dtype() -> None:
    array = np.array([{"value": 1}], dtype=object)

    with pytest.raises(TypeError):
        array_checksum(array)


def test_empty_array_uses_sha256_of_empty_bytes() -> None:
    array = np.array([], dtype=np.float32)

    assert array_checksum(array) == hashlib.sha256(b"").hexdigest()


def test_checksum_does_not_modify_input() -> None:
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    array = base[:, ::2]
    expected = array.copy()
    original_shape = array.shape
    original_dtype = array.dtype
    original_writeable = array.flags.writeable

    array_checksum(array)

    np.testing.assert_array_equal(array, expected)
    assert array.shape == original_shape
    assert array.dtype == original_dtype
    assert array.flags.writeable == original_writeable