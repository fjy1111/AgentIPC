import hashlib
from pathlib import Path

import pytest

from agentipc.artifacts.store import ArtifactStore
from agentipc.protocol.refs import ArtifactRef


def _ref_for_digest(digest: str, *, media_type: str = "application/octet-stream") -> ArtifactRef:
    return ArtifactRef(
        uri=f"artifact://sha256/{digest}",
        sha256=digest,
        media_type=media_type,
        size_bytes=0,
        summary="missing",
    )


def test_put_bytes_returns_expected_artifact_ref(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    data = "hello, 世界".encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()

    ref = store.put_bytes(
        data,
        media_type=" text/plain ",
        summary=" original summary ",
    )

    assert isinstance(ref, ArtifactRef)
    assert ref.sha256 == digest
    assert ref.uri == f"artifact://sha256/{digest}"
    assert ref.media_type == " text/plain "
    assert ref.summary == " original summary "
    assert ref.size_bytes == len(data)
    assert store._path_for_digest(digest).read_bytes() == data


def test_put_get_bytes_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    data = b"\x00\x01artifact payload\xff"

    ref = store.put_bytes(data, media_type="application/octet-stream", summary="payload")

    assert store.get_bytes(ref) == data
    assert store.exists(ref) is True


def test_empty_bytes_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    ref = store.put_bytes(b"", media_type="application/octet-stream", summary="")

    assert ref.sha256 == hashlib.sha256(b"").hexdigest()
    assert ref.size_bytes == 0
    assert store.get_bytes(ref) == b""


def test_same_bytes_share_identity_but_keep_call_metadata(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    data = b"same payload"

    first = store.put_bytes(data, media_type="text/plain", summary="first")
    second = store.put_bytes(data, media_type="application/custom", summary="second")

    assert first.sha256 == second.sha256
    assert first.uri == second.uri
    assert store._path_for_digest(first.sha256) == store._path_for_digest(second.sha256)
    assert first.media_type == "text/plain"
    assert first.summary == "first"
    assert second.media_type == "application/custom"
    assert second.summary == "second"


def test_different_bytes_have_different_digests(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    first = store.put_bytes(b"one", media_type="text/plain", summary="one")
    second = store.put_bytes(b"two", media_type="text/plain", summary="two")

    assert first.sha256 != second.sha256
    assert first.uri != second.uri


def test_missing_valid_ref_exists_false_and_get_raises(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = _ref_for_digest("0" * 64)

    assert store.exists(ref) is False
    with pytest.raises(FileNotFoundError):
        store.get_bytes(ref)


@pytest.mark.parametrize("data", ["text", bytearray(b"x"), memoryview(b"x"), [120]])
def test_put_bytes_rejects_non_bytes(tmp_path: Path, data: object) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(TypeError):
        store.put_bytes(data, media_type="text/plain", summary="x")  # type: ignore[arg-type]


def test_put_bytes_rejects_empty_media_type(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError):
        store.put_bytes(b"x", media_type="", summary="x")


def test_put_bytes_rejects_non_string_media_type(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(TypeError):
        store.put_bytes(b"x", media_type=123, summary="x")  # type: ignore[arg-type]


def test_put_bytes_rejects_non_string_summary(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(TypeError):
        store.put_bytes(b"x", media_type="text/plain", summary=None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "uri",
    [
        "file:///tmp/payload",
        f"artifact://other/{'a' * 64}",
        "artifact://sha256/",
        f"artifact://sha256/{'a' * 64}/extra",
        f"artifact://sha256/{'a' * 64}?x=1",
        f"artifact://sha256/{'a' * 64}#fragment",
        "artifact://sha256/../../etc/passwd",
        "artifact://sha256/%2Fetc%2Fpasswd",
    ],
)
def test_transport_style_malformed_uri_is_rejected(tmp_path: Path, uri: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = ArtifactRef(
        uri=uri,
        sha256="a" * 64,
        media_type="application/octet-stream",
        size_bytes=0,
        summary="bad",
    )

    with pytest.raises(ValueError):
        store.exists(ref)
    with pytest.raises(ValueError):
        store.get_bytes(ref)


def test_uri_digest_must_match_ref_sha256(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    ref = ArtifactRef(
        uri=f"artifact://sha256/{'a' * 64}",
        sha256="b" * 64,
        media_type="application/octet-stream",
        size_bytes=0,
        summary="mismatch",
    )

    with pytest.raises(ValueError):
        store.exists(ref)
    with pytest.raises(ValueError):
        store.get_bytes(ref)


def test_uppercase_ref_digest_resolves_lowercase_path(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    original = store.put_bytes(b"case", media_type="text/plain", summary="case")
    uppercase = ArtifactRef(
        uri=f"artifact://sha256/{original.sha256.upper()}",
        sha256=original.sha256.upper(),
        media_type=original.media_type,
        size_bytes=original.size_bytes,
        summary=original.summary,
    )

    assert store.get_bytes(uppercase) == b"case"
    assert store.exists(uppercase) is True
    assert store._path_for_digest(uppercase.sha256) == store._path_for_digest(original.sha256)


def test_get_and_exists_reject_non_artifact_ref(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(TypeError):
        store.get_bytes("not-a-ref")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        store.exists("not-a-ref")  # type: ignore[arg-type]