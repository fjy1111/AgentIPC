import hashlib
from pathlib import Path

from agentipc.artifacts.store import ArtifactStore


LARGE_PAYLOAD_SIZE = 1024 * 1024 + 257


def _large_payload() -> bytes:
    pattern = b"AgentIPC-artifact-"
    repeats, remainder = divmod(LARGE_PAYLOAD_SIZE, len(pattern))
    return pattern * repeats + pattern[:remainder]


def test_bytes_round_trip_across_store_instances(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    payload = b"producer to consumer"
    producer = ArtifactStore(root)

    ref = producer.put_bytes(
        payload,
        media_type="application/octet-stream",
        summary="cross store bytes",
    )

    consumer = ArtifactStore(root)
    assert consumer.get_bytes(ref) == payload


def test_json_round_trip_across_store_instances(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    value = {"task": "AgentIPC", "items": [1, 2, "中文"], "ok": True}
    producer = ArtifactStore(root)

    ref = producer.put_json(value, summary="cross store json")

    consumer = ArtifactStore(root)
    assert consumer.get_json(ref) == value


def test_repeated_write_keeps_single_content_identity(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = ArtifactStore(root)
    payload = b"deduplicated payload"

    refs = [
        store.put_bytes(payload, media_type="text/plain", summary=f"write {index}")
        for index in range(3)
    ]

    assert len({ref.sha256 for ref in refs}) == 1
    assert len({ref.uri for ref in refs}) == 1
    assert len(list((root / "sha256").iterdir())) == 1


def test_empty_payload_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    ref = store.put_bytes(b"", media_type="application/octet-stream", summary="empty")

    assert ref.sha256 == hashlib.sha256(b"").hexdigest()
    assert ref.size_bytes == 0
    assert store.exists(ref) is True
    assert store.get_bytes(ref) == b""


def test_large_payload_round_trip(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")
    payload = _large_payload()
    expected_digest = hashlib.sha256(payload).hexdigest()

    ref = store.put_bytes(
        payload,
        media_type="application/octet-stream",
        summary="large payload",
    )

    assert len(payload) == LARGE_PAYLOAD_SIZE
    assert ref.sha256 == expected_digest
    assert ref.uri == f"artifact://sha256/{expected_digest}"
    assert ref.size_bytes == LARGE_PAYLOAD_SIZE
    assert store.get_bytes(ref) == payload