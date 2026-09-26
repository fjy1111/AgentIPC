import numpy as np
import pytest

from agentipc.state.inproc import InProcessArrayStore


def test_put_get_round_trip() -> None:
    store = InProcessArrayStore()
    array = np.arange(12, dtype=np.float32).reshape(3, 4)

    key = store.put(array)
    restored = store.get(key)

    np.testing.assert_array_equal(restored, array)
    assert restored.shape == array.shape
    assert restored.dtype == array.dtype


def test_put_generates_unique_keys() -> None:
    store = InProcessArrayStore()
    array = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    first_key = store.put(array)
    second_key = store.put(array)

    assert first_key != second_key


def test_put_stores_independent_copy() -> None:
    store = InProcessArrayStore()
    original = np.array([1, 2, 3], dtype=np.int32)
    expected = original.copy()

    key = store.put(original)
    original[...] = 99

    restored = store.get(key)
    np.testing.assert_array_equal(restored, expected)


def test_get_returns_independent_copy() -> None:
    store = InProcessArrayStore()
    key = store.put(np.array([1, 2, 3], dtype=np.int32))

    first = store.get(key)
    first[...] = 99
    second = store.get(key)

    np.testing.assert_array_equal(
        second,
        np.array([1, 2, 3], dtype=np.int32),
    )


def test_non_contiguous_input_round_trip() -> None:
    store = InProcessArrayStore()
    base = np.arange(12, dtype=np.float32).reshape(3, 4)
    array = base[:, ::2]
    assert not array.flags.c_contiguous

    key = store.put(array)
    restored = store.get(key)

    np.testing.assert_array_equal(restored, array)
    assert restored.shape == array.shape
    assert restored.dtype == array.dtype


def test_put_rejects_object_dtype() -> None:
    store = InProcessArrayStore()
    array = np.array([{"value": 1}], dtype=object)

    with pytest.raises(TypeError):
        store.put(array)


def test_put_rejects_empty_array() -> None:
    store = InProcessArrayStore()
    array = np.array([], dtype=np.float32)

    with pytest.raises(ValueError):
        store.put(array)


def test_put_rejects_python_list() -> None:
    store = InProcessArrayStore()

    with pytest.raises(TypeError):
        store.put([1.0, 2.0, 3.0])


def test_release_removes_entry() -> None:
    store = InProcessArrayStore()
    key = store.put(np.array([1.0], dtype=np.float32))

    store.release(key)

    with pytest.raises(KeyError):
        store.get(key)


def test_duplicate_release_is_idempotent() -> None:
    store = InProcessArrayStore()
    key = store.put(np.array([1.0], dtype=np.float32))

    store.release(key)
    store.release(key)


def test_store_instances_are_isolated() -> None:
    first_store = InProcessArrayStore()
    second_store = InProcessArrayStore()
    key = first_store.put(np.array([1.0], dtype=np.float32))

    with pytest.raises(KeyError):
        second_store.get(key)