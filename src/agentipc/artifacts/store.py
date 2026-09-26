import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agentipc.protocol.refs import ArtifactRef


_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if self.root.exists() and not self.root.is_dir():
            raise NotADirectoryError(str(self.root))

        self._sha256_root = self.root / "sha256"
        self._sha256_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_digest(digest: str) -> str:
        if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("digest must be exactly 64 hexadecimal characters")
        return digest.lower()

    def _path_for_digest(self, digest: str) -> Path:
        normalized = self._normalize_digest(digest)
        return self._sha256_root / normalized

    def _digest_from_ref(self, ref: ArtifactRef) -> str:
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")

        parsed = urlsplit(ref.uri)
        if (
            parsed.scheme != "artifact"
            or parsed.netloc != "sha256"
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("invalid artifact URI")

        match = re.fullmatch(r"/([0-9a-fA-F]{64})", parsed.path)
        if match is None:
            raise ValueError("invalid artifact URI")

        uri_digest = self._normalize_digest(match.group(1))
        ref_digest = self._normalize_digest(ref.sha256)
        if uri_digest != ref_digest:
            raise ValueError("artifact URI digest does not match ref.sha256")
        return uri_digest

    def put_bytes(
        self,
        data: bytes,
        *,
        media_type: str,
        summary: str,
    ) -> ArtifactRef:
        if not isinstance(data, bytes):
            raise TypeError("data must be bytes")
        if not isinstance(media_type, str):
            raise TypeError("media_type must be a str")
        if media_type == "":
            raise ValueError("media_type must not be empty")
        if not isinstance(summary, str):
            raise TypeError("summary must be a str")

        digest = hashlib.sha256(data).hexdigest()
        self._path_for_digest(digest).write_bytes(data)

        return ArtifactRef(
            uri=f"artifact://sha256/{digest}",
            sha256=digest,
            media_type=media_type,
            size_bytes=len(data),
            summary=summary,
        )

    def put_json(
        self,
        value: Any,
        *,
        summary: str,
    ) -> ArtifactRef:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return self.put_bytes(
            payload,
            media_type="application/json",
            summary=summary,
        )

    def get_bytes(self, ref: ArtifactRef) -> bytes:
        digest = self._digest_from_ref(ref)
        payload = self._path_for_digest(digest).read_bytes()
        actual_digest = hashlib.sha256(payload).hexdigest()
        if actual_digest != digest:
            raise ValueError("artifact sha256 integrity check failed")
        return payload

    def get_json(self, ref: ArtifactRef) -> Any:
        if not isinstance(ref, ArtifactRef):
            raise TypeError("ref must be an ArtifactRef")
        if ref.media_type != "application/json":
            raise ValueError("artifact media_type must be application/json")

        payload = self.get_bytes(ref)
        return json.loads(payload.decode("utf-8"))

    def exists(self, ref: ArtifactRef) -> bool:
        digest = self._digest_from_ref(ref)
        return self._path_for_digest(digest).is_file()