from pathlib import Path

import pytest

from agentipc.artifacts.store import ArtifactStore
from agentipc.protocol.refs import ArtifactRef


def _payload_path(store: ArtifactStore, sha256: str) -> Path:
    return store.root / "sha256" / sha256.lower()


def test_valid_artifact_still_reads_after_integrity_validation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    payload = b"valid artifact payload"

    ref = store.put_bytes(
        payload,
        media_type="application/octet-stream",
        summary="valid",
    )

    assert store.get_bytes(ref) == payload


def test_same_size_corruption_is_detected_by_sha256(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(b"abcdef", media_type="text/plain", summary="same size")
    _payload_path(store, ref.sha256).write_bytes(b"abcdeg")

    with pytest.raises(ValueError, match="(?i)(sha256|integrity)"):
        store.get_bytes(ref)


def test_changed_size_corruption_is_detected_by_sha256(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(
        b"artifact payload",
        media_type="application/octet-stream",
        summary="truncated",
    )
    _payload_path(store, ref.sha256).write_bytes(b"artifact")

    with pytest.raises(ValueError, match="(?i)(sha256|integrity)"):
        store.get_bytes(ref)


def test_get_json_detects_corruption_before_utf8_decode(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_json({"answer": "ok"}, summary="json")
    _payload_path(store, ref.sha256).write_bytes(b"\xff\xfe\xfd")

    with pytest.raises(ValueError, match="(?i)(sha256|integrity)"):
        store.get_json(ref)


def test_exists_remains_true_for_corrupt_present_artifact(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(b"original", media_type="text/plain", summary="corrupt")
    _payload_path(store, ref.sha256).write_bytes(b"tampered")

    assert store.exists(ref) is True
    with pytest.raises(ValueError, match="(?i)(sha256|integrity)"):
        store.get_bytes(ref)


def test_missing_artifact_remains_file_not_found(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = store.put_bytes(b"remove me", media_type="text/plain", summary="missing")
    _payload_path(store, ref.sha256).unlink()

    assert store.exists(ref) is False
    with pytest.raises(FileNotFoundError):
        store.get_bytes(ref)


def test_uppercase_ref_passes_integrity_validation(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    payload = b"uppercase digest"
    original = store.put_bytes(payload, media_type="text/plain", summary="uppercase")
    uppercase = ArtifactRef(
        uri=f"artifact://sha256/{original.sha256.upper()}",
        sha256=original.sha256.upper(),
        media_type=original.media_type,
        size_bytes=original.size_bytes,
        summary=original.summary,
    )

    assert store.get_bytes(uppercase) == payload