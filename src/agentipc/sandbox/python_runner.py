from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from agentipc.sandbox.models import SandboxResult


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
                text=True,
                encoding="utf-8",
                errors="replace",
            )

            timed_out = False
            try:
                stdout, stderr = process.communicate(timeout=normalized_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                stdout, stderr = process.communicate()

            duration_ms = (time.perf_counter() - started) * 1000.0
            if process.returncode is None:
                raise RuntimeError("subprocess returncode is unavailable after communicate")

            return SandboxResult(
                exit_code=process.returncode,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                duration_ms=duration_ms,
            )
