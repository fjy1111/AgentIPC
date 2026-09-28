from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from typing import BinaryIO

from agentipc.sandbox.models import SandboxResult


_OUTPUT_LIMIT_BYTES = 64 * 1024
_READ_CHUNK_BYTES = 8192
_TRUNCATION_MARKER = "\n...[AgentIPC output truncated]...\n"


class _BoundedStreamCapture:
    def __init__(self) -> None:
        self._retained = bytearray()
        self.truncated = False
        self.error: BaseException | None = None

    def drain(self, pipe: BinaryIO) -> None:
        try:
            while True:
                chunk = pipe.read(_READ_CHUNK_BYTES)
                if not chunk:
                    break

                remaining = _OUTPUT_LIMIT_BYTES - len(self._retained)
                if remaining > 0:
                    self._retained.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    self.truncated = True
        except BaseException as exc:
            self.error = exc
        finally:
            pipe.close()

    def decode(self) -> str:
        text = bytes(self._retained).decode("utf-8", errors="replace")
        if self.truncated:
            return text + _TRUNCATION_MARKER
        return text


class PythonSandbox:
    def run(
        self,
        code: str,
        *,
        timeout_sec: float,
    ) -> SandboxResult:
        if type(code) is not str:
            raise TypeError("code must be a str")

        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)):
            raise TypeError("timeout_sec must be an int or float")
        normalized_timeout = float(timeout_sec)
        if not math.isfinite(normalized_timeout):
            raise ValueError("timeout_sec must be finite")
        if normalized_timeout <= 0:
            raise ValueError("timeout_sec must be greater than 0")

        with tempfile.TemporaryDirectory(prefix="agentipc-sandbox-") as workspace:
            script_path = Path(workspace) / "main.py"
            script_path.write_text(code, encoding="utf-8")

            started = time.perf_counter()
            process = subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=workspace,
                shell=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if process.stdout is None or process.stderr is None:
                process.kill()
                process.wait()
                raise RuntimeError("subprocess output pipes were not created")

            stdout_capture = _BoundedStreamCapture()
            stderr_capture = _BoundedStreamCapture()
            stdout_reader = threading.Thread(
                target=stdout_capture.drain,
                args=(process.stdout,),
                daemon=True,
            )
            stderr_reader = threading.Thread(
                target=stderr_capture.drain,
                args=(process.stderr,),
                daemon=True,
            )
            readers = (stdout_reader, stderr_reader)

            try:
                for reader in readers:
                    reader.start()
            except BaseException:
                process.kill()
                process.wait()
                for reader in readers:
                    if reader.ident is not None:
                        reader.join()
                raise

            timed_out = False
            try:
                process.wait(timeout=normalized_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()

            for reader in readers:
                reader.join()

            duration_ms = (time.perf_counter() - started) * 1000.0
            if process.returncode is None:
                raise RuntimeError("subprocess returncode is unavailable after wait")

            for stream_name, capture in (
                ("stdout", stdout_capture),
                ("stderr", stderr_capture),
            ):
                if capture.error is not None:
                    raise RuntimeError(f"{stream_name} reader failed") from capture.error

            return SandboxResult(
                exit_code=process.returncode,
                stdout=stdout_capture.decode(),
                stderr=stderr_capture.decode(),
                timed_out=timed_out,
                duration_ms=duration_ms,
            )
