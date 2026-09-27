import math

import pytest

from agentipc.memory.scoring import keyword_overlap_score


def test_exact_full_match() -> None:
    assert keyword_overlap_score(["linux", "ipc"], ["linux", "ipc"]) == 1.0


def test_no_match() -> None:
    assert keyword_overlap_score(["linux"], ["windows"]) == 0.0


def test_partial_query_coverage() -> None:
    assert keyword_overlap_score(["linux", "ipc", "memory"], ["linux", "memory"]) == pytest.approx(2 / 3)


def test_denominator_is_unique_query_count() -> None:
    assert keyword_overlap_score(["linux", "linux", "ipc"], ["linux"]) == 0.5


def test_case_insensitive() -> None:
    assert keyword_overlap_score(["LiNuX"], ["linux"]) == 1.0


def test_nfkc_equivalent_strings() -> None:
    assert keyword_overlap_score(["Ｌｉｎｕｘ"], ["Linux"]) == 1.0


def test_surrounding_whitespace_is_ignored() -> None:
    assert keyword_overlap_score(["  linux\t"], ["linux"]) == 1.0


def test_duplicate_query_keyword_does_not_change_denominator() -> None:
    assert keyword_overlap_score(["Linux", "linux", "LINUX"], ["linux"]) == 1.0


def test_duplicate_record_keyword_does_not_increase_score() -> None:
    assert keyword_overlap_score(["linux", "ipc"], ["linux", "Linux", "LINUX"]) == 0.5


def test_extra_record_keywords_are_not_penalized() -> None:
    assert keyword_overlap_score(
        ["linux", "shared memory"],
        ["linux", "shared memory", "ipc", "state", "python"],
    ) == 1.0


def test_exact_term_overlap_does_not_use_substrings() -> None:
    assert keyword_overlap_score(["linux"], ["linux kernel"]) == 0.0


def test_empty_query_list_returns_zero() -> None:
    assert keyword_overlap_score([], ["linux"]) == 0.0


def test_query_with_only_normalized_empty_terms_returns_zero() -> None:
    assert keyword_overlap_score(["", " ", "\t"], ["linux"]) == 0.0


def test_normalized_empty_terms_are_ignored_when_valid_terms_exist() -> None:
    assert keyword_overlap_score(["", " ", "Linux"], ["linux"]) == 1.0


def test_empty_record_returns_zero() -> None:
    assert keyword_overlap_score(["linux"], []) == 0.0


def test_both_empty_returns_zero() -> None:
    assert keyword_overlap_score([], []) == 0.0


@pytest.mark.parametrize("bad", [("linux",), {"linux"}, "linux", iter(["linux"]), None])
def test_non_list_query_raises_type_error(bad) -> None:
    with pytest.raises(TypeError):
        keyword_overlap_score(bad, ["linux"])


@pytest.mark.parametrize("bad", [("linux",), {"linux"}, "linux", iter(["linux"]), None])
def test_non_list_record_raises_type_error(bad) -> None:
    with pytest.raises(TypeError):
        keyword_overlap_score(["linux"], bad)


@pytest.mark.parametrize("query,record", [(["linux", 1], ["linux"]), (["linux"], ["linux", None])])
def test_non_string_item_raises_type_error(query, record) -> None:
    with pytest.raises(TypeError):
        keyword_overlap_score(query, record)


def test_result_is_finite_float_in_unit_interval() -> None:
    score = keyword_overlap_score(["linux", "ipc", "state"], ["linux", "state"])
    assert isinstance(score, float)
    assert math.isfinite(score)
    assert 0.0 <= score <= 1.0
