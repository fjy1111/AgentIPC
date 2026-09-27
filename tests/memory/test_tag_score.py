import math

import pytest

from agentipc.memory.scoring import tag_overlap_score


def test_full_tag_overlap() -> None:
    assert tag_overlap_score(["linux", "memory"], ["linux", "memory"]) == 1.0


def test_no_tag_overlap() -> None:
    assert tag_overlap_score(["linux"], ["database"]) == 0.0


def test_partial_tag_overlap() -> None:
    assert tag_overlap_score(["linux", "ipc", "memory"], ["linux", "ipc"]) == pytest.approx(2 / 3)


def test_tag_casefold() -> None:
    assert tag_overlap_score(["MEMORY"], ["memory"]) == 1.0


def test_tag_nfkc_normalization() -> None:
    assert tag_overlap_score(["Ｍｅｍｏｒｙ"], ["Memory"]) == 1.0


def test_tag_whitespace_normalization() -> None:
    assert tag_overlap_score(["  memory \n"], ["memory"]) == 1.0


def test_duplicate_query_tag_does_not_add_weight() -> None:
    assert tag_overlap_score(["memory", "MEMORY", "ipc"], ["memory"]) == 0.5


def test_duplicate_record_tag_does_not_add_weight() -> None:
    assert tag_overlap_score(["memory", "ipc"], ["memory", "Memory", "MEMORY"]) == 0.5


def test_extra_record_tags_are_not_penalized() -> None:
    assert tag_overlap_score(["memory"], ["memory", "linux", "ipc"]) == 1.0


def test_empty_query_returns_zero() -> None:
    assert tag_overlap_score([], ["memory"]) == 0.0


def test_empty_record_returns_zero() -> None:
    assert tag_overlap_score(["memory"], []) == 0.0


def test_normalized_empty_strings_are_ignored() -> None:
    assert tag_overlap_score(["", "  ", "memory"], ["\t", "memory"]) == 1.0


@pytest.mark.parametrize("bad", [("memory",), {"memory"}, "memory", iter(["memory"]), None])
def test_invalid_query_container_raises_type_error(bad) -> None:
    with pytest.raises(TypeError):
        tag_overlap_score(bad, ["memory"])


@pytest.mark.parametrize("bad", [("memory",), {"memory"}, "memory", iter(["memory"]), None])
def test_invalid_record_container_raises_type_error(bad) -> None:
    with pytest.raises(TypeError):
        tag_overlap_score(["memory"], bad)


@pytest.mark.parametrize("query,record", [(["memory", 1], ["memory"]), (["memory"], [None])])
def test_invalid_tag_member_raises_type_error(query, record) -> None:
    with pytest.raises(TypeError):
        tag_overlap_score(query, record)


def test_tag_result_is_finite_float_in_unit_interval() -> None:
    score = tag_overlap_score(["linux", "memory", "ipc"], ["linux", "ipc"])
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
