import json
from pathlib import Path

import pytest

from agentipc.artifacts.store import ArtifactStore


def test_unicode_nested_json_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    value = {
        "标题": "多智能体通信",
        "nested": {"emoji": "🤖", "items": [1, True, None, "中文"]},
    }

    ref = store.put_json(value, summary=" JSON 摘要 ")

    assert ref.media_type == "application/json"
    assert ref.summary == " JSON 摘要 "
    assert store.get_json(ref) == value


def test_put_json_is_deterministic_for_dict_insertion_order(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    first_value = {"z": 1, "a": {"y": 2, "x": 3}}
    second_value = {"a": {"x": 3, "y": 2}, "z": 1}

    first = store.put_json(first_value, summary="first")
    second = store.put_json(second_value, summary="second")

    expected = json.dumps(
        first_value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    assert first.sha256 == second.sha256
    assert first.uri == second.uri
    assert store.get_bytes(first) == expected


@pytest.mark.parametrize(
    "value",
    [
        ["a", 1, False, None],
        "scalar 中文",
        42,
        3.25,
        True,
        None,
    ],
)
def test_list_and_scalar_json_round_trip(tmp_path: Path, value: object) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    ref = store.put_json(value, summary="value")

    assert store.get_json(ref) == value


def test_stored_json_is_compact_utf8_without_ascii_escaping(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    value = {"z": "中文", "a": "é"}

    ref = store.put_json(value, summary="unicode")
    payload = store.get_bytes(ref)

    assert payload == '{"a":"é","z":"中文"}'.encode("utf-8")
    assert b"\\u" not in payload


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_json_numbers_are_rejected(tmp_path: Path, value: float) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError):
        store.put_json({"value": value}, summary="bad number")


def test_unsupported_json_object_is_rejected(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(TypeError):
        store.put_json({"value": object()}, summary="unsupported")


def test_get_json_rejects_non_json_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(b"plain text", media_type="text/plain", summary="plain")

    with pytest.raises(ValueError):
        store.get_json(ref)


def test_get_json_rejects_non_artifact_ref(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(TypeError):
        store.get_json("not-a-ref")  # type: ignore[arg-type]