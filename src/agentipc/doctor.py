from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util as importlib_util
import json
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import sqlite3
import sys
import tempfile
from typing import Callable, Literal

from agentipc.providers.hash_embedding import HashEmbeddingProvider
from agentipc.providers.mock_llm import MockLLMProvider


DoctorStatus = Literal["PASS", "FAIL", "WARN"]


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    name: str
    status: DoctorStatus
    required: bool
    message: str


@dataclass(frozen=True, slots=True)
class DoctorReport:
    ok: bool
    checks: tuple[DoctorCheck, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
        }


def _pass(name: str, message: str, *, required: bool) -> DoctorCheck:
    return DoctorCheck(name=name, status="PASS", required=required, message=message)


def _fail(name: str, message: str, *, required: bool) -> DoctorCheck:
    return DoctorCheck(name=name, status="FAIL", required=required, message=message)


def _warn(name: str, message: str) -> DoctorCheck:
    return DoctorCheck(name=name, status="WARN", required=False, message=message)


def _check_python() -> DoctorCheck:
    version = sys.version_info
    actual = f"{version.major}.{version.minor}.{version.micro}"
    if (version.major, version.minor) >= (3, 10):
        return _pass("python", f"Python {actual} (>= 3.10)", required=True)
    return _fail("python", f"Python {actual}; requires >= 3.10", required=True)


def _check_filesystem() -> DoctorCheck:
    try:
        with tempfile.TemporaryDirectory(
            prefix=".agentipc-doctor-",
            dir=Path.cwd(),
        ) as temp_dir:
            path = Path(temp_dir) / "doctor.tmp"
            payload = "agentipc doctor filesystem round-trip"
            path.write_text(payload, encoding="utf-8")
            if path.read_text(encoding="utf-8") != payload:
                raise RuntimeError("temporary file round-trip mismatch")
    except Exception as exc:
        return _fail(
            "filesystem",
            f"current directory temporary file check failed: {exc}",
            required=True,
        )
    return _pass(
        "filesystem",
        "current directory supports temporary create/write/read/cleanup",
        required=True,
    )


def _check_sqlite() -> DoctorCheck:
    connection: sqlite3.Connection | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix=".agentipc-doctor-sqlite-",
            dir=Path.cwd(),
        ) as temp_dir:
            db_path = Path(temp_dir) / "doctor.sqlite3"
            connection = sqlite3.connect(db_path)
            with connection:
                connection.execute("CREATE TABLE probe (value TEXT NOT NULL)")
                connection.execute("INSERT INTO probe(value) VALUES (?)", ("ok",))
            row = connection.execute("SELECT value FROM probe").fetchone()
            if row != ("ok",):
                raise RuntimeError("SQLite round-trip mismatch")
            connection.close()
            connection = None
    except Exception as exc:
        return _fail(
            "sqlite",
            f"SQLite create/insert/select check failed: {exc}",
            required=True,
        )
    finally:
        if connection is not None:
            connection.close()
    return _pass(
        "sqlite",
        "SQLite create/insert/select round-trip succeeded",
        required=True,
    )


def _check_shared_memory() -> DoctorCheck:
    owner: SharedMemory | None = None
    attached: SharedMemory | None = None
    payload = b"agentipc-doctor-shm"
    try:
        owner = SharedMemory(create=True, size=len(payload))
        owner.buf[: len(payload)] = payload
        attached = SharedMemory(name=owner.name)
        observed = bytes(attached.buf[: len(payload)])
        if observed != payload:
            raise RuntimeError("shared memory round-trip mismatch")
    except Exception as exc:
        return _fail(
            "shared_memory",
            f"shared memory create/attach/read check failed: {exc}",
            required=True,
        )
    finally:
        if attached is not None:
            attached.close()
        if owner is not None:
            owner.close()
            try:
                owner.unlink()
            except FileNotFoundError:
                pass
    return _pass(
        "shared_memory",
        "shared memory create/write/attach/read/cleanup succeeded",
        required=True,
    )


def _check_mock_provider() -> DoctorCheck:
    try:
        provider = MockLLMProvider(default_text="doctor-ok")
        response = provider.complete(
            [{"role": "user", "content": "offline doctor probe"}],
            temperature=0.0,
        )
        if response.text != "doctor-ok":
            raise RuntimeError("unexpected mock provider response")
    except Exception as exc:
        return _fail(
            "provider_mock",
            f"built-in MockLLMProvider check failed: {exc}",
            required=True,
        )
    return _pass(
        "provider_mock",
        "built-in MockLLMProvider is available offline",
        required=True,
    )


def _check_hash_embedding() -> DoctorCheck:
    try:
        provider = HashEmbeddingProvider(dim=64)
        matrix = provider.embed(["agentipc doctor"])
        if matrix.shape != (1, 64):
            raise RuntimeError(f"unexpected embedding shape: {matrix.shape}")
    except Exception as exc:
        return _fail(
            "provider_hash_embedding",
            f"built-in HashEmbeddingProvider check failed: {exc}",
            required=True,
        )
    return _pass(
        "provider_hash_embedding",
        "built-in HashEmbeddingProvider(dim=64) is available offline",
        required=True,
    )


def _check_optional_dependency(display_name: str, module_name: str) -> DoctorCheck:
    try:
        available = importlib_util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return _warn(
            f"optional_{module_name}",
            f"optional dependency {display_name} could not be inspected: {exc}",
        )

    if available:
        return _pass(
            f"optional_{module_name}",
            f"optional dependency {display_name} is installed",
            required=False,
        )
    return _warn(
        f"optional_{module_name}",
        f"optional dependency {display_name} is not installed",
    )


def run_doctor() -> DoctorReport:
    core_checks: tuple[Callable[[], DoctorCheck], ...] = (
        _check_python,
        _check_filesystem,
        _check_sqlite,
        _check_shared_memory,
        _check_mock_provider,
        _check_hash_embedding,
    )
    checks = [check() for check in core_checks]
    checks.extend(
        [
            _check_optional_dependency("openai", "openai"),
            _check_optional_dependency(
                "sentence-transformers",
                "sentence_transformers",
            ),
            _check_optional_dependency("tiktoken", "tiktoken"),
        ]
    )
    ok = all(
        check.status == "PASS"
        for check in checks
        if check.required
    )
    return DoctorReport(ok=ok, checks=tuple(checks))


def render_doctor_json(report: DoctorReport) -> str:
    if not isinstance(report, DoctorReport):
        raise TypeError("report must be a DoctorReport")
    return json.dumps(
        report.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def render_doctor_human(report: DoctorReport) -> str:
    if not isinstance(report, DoctorReport):
        raise TypeError("report must be a DoctorReport")

    lines = ["AgentIPC doctor"]
    for check in report.checks:
        requirement = "required" if check.required else "optional"
        lines.append(
            f"{check.status:4} {check.name} [{requirement}] - {check.message}"
        )
    lines.append("Overall: PASS" if report.ok else "Overall: FAIL")
    return "\n".join(lines)
