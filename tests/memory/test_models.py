import math

import pytest
from pydantic import ValidationError

from agentipc.memory.models import MemoryRecord, MemoryType


def make_record(**overrides) -> MemoryRecord:
    values = {
        "memory_id": "mem-1",
        "source_agent": "planner",
        "task_topic": "topic",
        "summary": "summary",
        "memory_type": MemoryType.EVIDENCE,
    }
    values.update(overrides)
    return MemoryRecord(**values)


def test_memory_type_has_exactly_three_values() -> None:
    assert {item.value for item in MemoryType} == {
        "evidence",
        "experience",
        "result",
    }
    assert len(MemoryType) == 3


def test_minimal_record_uses_locked_defaults() -> None:
    record = make_record()

    assert record.memory_id == "mem-1"
    assert record.source_agent == "planner"
    assert record.task_topic == "topic"
    assert record.summary == "summary"
    assert record.memory_type is MemoryType.EVIDENCE
    assert math.isfinite(record.created_at)
    assert record.created_at >= 0
    assert record.tags == []
    assert record.keywords == []
    assert record.embedding is None
    assert record.payload == {}
    assert record.reuse_count == 0
    assert record.success_count == 0
    assert record.failure_count == 0
    assert record.last_accessed_at is None


def test_explicit_created_at_is_preserved_as_float() -> None:
    record = make_record(created_at=123)
    assert record.created_at == 123.0
    assert isinstance(record.created_at, float)


def test_mutable_defaults_are_not_shared() -> None:
    first = make_record(memory_id="first")
    second = make_record(memory_id="second")

    first.tags.append("tag")
    first.keywords.append("key")
    first.payload["nested"] = {"value": 1}

    assert second.tags == []
    assert second.keywords == []
    assert second.payload == {}


@pytest.mark.parametrize("memory_type", list(MemoryType))
def test_all_memory_types_are_valid(memory_type: MemoryType) -> None:
    assert make_record(memory_type=memory_type).memory_type is memory_type


@pytest.mark.parametrize("field", ["memory_id", "source_agent", "task_topic", "summary"])
def test_required_text_fields_reject_empty_string(field: str) -> None:
    with pytest.raises(ValidationError):
        make_record(**{field: ""})


def test_required_text_fields_preserve_whitespace() -> None:
    record = make_record(
        memory_id=" id ",
        source_agent=" agent ",
        task_topic=" topic ",
        summary=" summary ",
    )
    assert record.memory_id == " id "
    assert record.source_agent == " agent "
    assert record.task_topic == " topic "
    assert record.summary == " summary "


@pytest.mark.parametrize("created_at", [-1.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_created_at_is_rejected(created_at: float) -> None:
    with pytest.raises(ValidationError):
        make_record(created_at=created_at)


def test_tags_require_a_list() -> None:
    with pytest.raises(ValidationError):
        make_record(tags=("a", "b"))


def test_tags_require_strings() -> None:
    with pytest.raises(ValidationError):
        make_record(tags=["a", 1])


def test_empty_tag_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_record(tags=[""])


def test_keywords_require_a_list() -> None:
    with pytest.raises(ValidationError):
        make_record(keywords=("a", "b"))


def test_keywords_require_strings() -> None:
    with pytest.raises(ValidationError):
        make_record(keywords=["a", 1])


def test_empty_keyword_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_record(keywords=[""])


def test_tags_and_keywords_are_not_normalized() -> None:
    record = make_record(
        tags=[" Tag ", "Tag", "Tag"],
        keywords=["Linux", "linux", "Linux"],
    )
    assert record.tags == [" Tag ", "Tag", "Tag"]
    assert record.keywords == ["Linux", "linux", "Linux"]


def test_finite_embedding_list_is_accepted() -> None:
    record = make_record(embedding=[0, 1.5, -2])
    assert record.embedding == [0.0, 1.5, -2.0]


def test_none_embedding_is_accepted() -> None:
    assert make_record(embedding=None).embedding is None


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_embedding_item_is_rejected(bad: float) -> None:
    with pytest.raises(ValidationError):
        make_record(embedding=[0.0, bad])


@pytest.mark.parametrize("field", ["reuse_count", "success_count", "failure_count"])
def test_negative_counter_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        make_record(**{field: -1})


@pytest.mark.parametrize("field", ["reuse_count", "success_count", "failure_count"])
@pytest.mark.parametrize("value", [True, False])
def test_bool_counter_is_rejected(field: str, value: bool) -> None:
    with pytest.raises(ValidationError):
        make_record(**{field: value})


@pytest.mark.parametrize(
    "last_accessed_at",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_last_accessed_at_is_rejected(last_accessed_at: float) -> None:
    with pytest.raises(ValidationError):
        make_record(last_accessed_at=last_accessed_at)


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_record(extra_field="nope")


def test_model_dump_validate_round_trip() -> None:
    original = make_record(
        created_at=100.25,
        tags=["one", "two"],
        keywords=["Linux", "linux"],
        embedding=[0.25, -0.5],
        payload={"nested": {"ok": True}},
        reuse_count=5,
        success_count=3,
        failure_count=1,
        last_accessed_at=120.5,
    )
    restored = MemoryRecord.model_validate(original.model_dump())
    assert restored == original


def test_unicode_metadata_and_payload_are_preserved() -> None:
    record = make_record(
        task_topic="共享记忆 🚀",
        summary="保留中文与 emoji ✅",
        tags=["中文", "🧠"],
        keywords=["记忆", "复用"],
        payload={"证据": {"结论": "可以复用 ✅"}},
    )
    assert record.task_topic == "共享记忆 🚀"
    assert record.summary == "保留中文与 emoji ✅"
    assert record.tags == ["中文", "🧠"]
    assert record.keywords == ["记忆", "复用"]
    assert record.payload == {"证据": {"结论": "可以复用 ✅"}}
