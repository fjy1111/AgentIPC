from pathlib import Path

import pytest

from agentipc.artifacts.store import ArtifactStore


def test_constructor_creates_root_and_sha256_directory(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"

    store = ArtifactStore(root)

    assert store.root == root
    assert root.is_dir()
    assert (root / "sha256").is_dir()


def test_constructor_creates_nested_root(tmp_path: Path) -> None:
    root = tmp_path / "nested" / "artifact" / "store"

    ArtifactStore(root)

    assert (root / "sha256").is_dir()


def test_constructor_can_reopen_existing_directory(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    ArtifactStore(root)

    reopened = ArtifactStore(root)

    assert reopened.root == root
    assert (root / "sha256").is_dir()


def test_constructor_rejects_root_that_is_a_file(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        ArtifactStore(root)


def test_digest_path_is_deterministic_and_lowercase(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    uppercase_digest = "ABCDEF" * 10 + "ABCD"
    expected = store.root / "sha256" / uppercase_digest.lower()

    first = store._path_for_digest(uppercase_digest)
    second = store._path_for_digest(uppercase_digest.lower())

    assert first == expected
    assert second == expected


@pytest.mark.parametrize(
    "digest",
    [
        "",
        "0" * 63,
        "0" * 65,
        "g" * 64,
        "../" + "0" * 61,
        "0" * 32 + "/" + "0" * 31,
    ],
)
def test_digest_path_rejects_invalid_digest(tmp_path: Path, digest: str) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError):
        store._path_for_digest(digest)


def test_digest_path_helper_does_not_create_payload(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    digest = "a" * 64

    path = store._path_for_digest(digest)

    assert path == store.root / "sha256" / digest
    assert not path.exists()