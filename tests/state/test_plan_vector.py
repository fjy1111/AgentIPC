from copy import deepcopy

import numpy as np
import pytest

from agentipc.state.plan_vector import PLAN_VECTOR_DIM, encode_plan_vector


def _sample_plan() -> dict[str, object]:
    return {
        "task_type": "log_analysis",
        "steps": [
            "retrieve_evidence",
            "inspect_logs",
            "summarize",
        ],
        "required_capabilities": [
            "retrieve",
            "execute",
            "summarize",
        ],
        "needs_tool": True,
    }


def test_same_plan_produces_exactly_same_vector() -> None:
    plan = _sample_plan()

    first = encode_plan_vector(plan)
    second = encode_plan_vector(plan)

    np.testing.assert_array_equal(first, second)


def test_mapping_insertion_order_does_not_change_vector() -> None:
    first_plan = {"a": 1, "b": 2, "c": {"x": True, "y": None}}
    second_plan = {"c": {"y": None, "x": True}, "b": 2, "a": 1}

    np.testing.assert_array_equal(
        encode_plan_vector(first_plan),
        encode_plan_vector(second_plan),
    )


def test_changed_important_field_changes_vector() -> None:
    first = _sample_plan()
    second = deepcopy(first)
    second["task_type"] = "code_generation"

    assert not np.array_equal(
        encode_plan_vector(first),
        encode_plan_vector(second),
    )


def test_changed_step_order_changes_vector() -> None:
    first = _sample_plan()
    second = deepcopy(first)
    second["steps"] = [
        "inspect_logs",
        "retrieve_evidence",
        "summarize",
    ]

    assert not np.array_equal(
        encode_plan_vector(first),
        encode_plan_vector(second),
    )


def test_nested_plan_and_unicode_are_supported() -> None:
    plan = {
        "任务": {
            "名称": "日志分析",
            "步骤": ("检索证据", {"动作": "总结", "启用": True}),
        },
        "备注": None,
    }

    vector = encode_plan_vector(plan)

    assert vector.shape == (PLAN_VECTOR_DIM,)
    assert np.all(np.isfinite(vector))


def test_vector_shape_dtype_layout_finiteness_and_normalization() -> None:
    vector = encode_plan_vector(_sample_plan())

    assert vector.shape == (PLAN_VECTOR_DIM,)
    assert vector.dtype == np.float32
    assert vector.flags.c_contiguous
    assert np.all(np.isfinite(vector))
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)


def test_bool_int_float_and_str_are_distinct() -> None:
    vectors = [
        encode_plan_vector({"value": value})
        for value in (True, 1, 1.0, "1")
    ]

    for left_index, left in enumerate(vectors):
        for right in vectors[left_index + 1 :]:
            assert not np.array_equal(left, right)


@pytest.mark.parametrize("value", [[], "plan", 1, 1.0, None])
def test_non_mapping_top_level_is_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="Mapping"):
        encode_plan_vector(value)  # type: ignore[arg-type]


def test_empty_top_level_plan_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        encode_plan_vector({})


def test_non_string_mapping_key_is_rejected() -> None:
    with pytest.raises(TypeError, match="keys"):
        encode_plan_vector({"nested": {1: "bad"}})  # type: ignore[dict-item]


def test_empty_mapping_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        encode_plan_vector({"nested": {"": "bad"}})


def test_set_is_rejected() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        encode_plan_vector({"steps": {"retrieve", "summarize"}})


def test_ndarray_is_rejected() -> None:
    with pytest.raises(TypeError, match="unsupported"):
        encode_plan_vector({"state": np.array([1.0], dtype=np.float32)})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_float_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        encode_plan_vector({"value": value})


def test_input_object_is_not_modified() -> None:
    plan = _sample_plan()
    before = deepcopy(plan)

    encode_plan_vector(plan)

    assert plan == before