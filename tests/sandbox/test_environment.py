from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agentipc.sandbox import limits
from agentipc.sandbox import python_runner
from agentipc.sandbox.python_runner import PythonSandbox


def _run_json(code: str) -> dict[str, object]:
    result = PythonSandbox().run(code, timeout_sec=2.0)
    assert result.exit_code == 0, result.stderr
    assert result.timed_out is False
    assert result.stderr == ""
    return json.loads(result.stdout)


def test_fake_secret_parent_variable_does_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIPC_TEST_SECRET", "should-not-leak")

    result = PythonSandbox().run(
        'import os\nprint(os.environ.get("AGENTIPC_TEST_SECRET"))',
        timeout_sec=2.0,
    )

    assert result.exit_code == 0
    assert result.stdout == "None\n"
    assert os.environ["AGENTIPC_TEST_SECRET"] == "should-not-leak"


def test_unknown_parent_variable_is_default_denied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTIPC_ARBITRARY_PARENT_VALUE", "parent-only")

    result = PythonSandbox().run(
        'import os\nprint(os.environ.get("AGENTIPC_ARBITRARY_PARENT_VALUE"))',
        timeout_sec=2.0,
    )

    assert result.exit_code == 0
    assert result.stdout == "None\n"
    assert os.environ["AGENTIPC_ARBITRARY_PARENT_VALUE"] == "parent-only"


def test_python_injection_variables_do_not_leak(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/malicious/test/path")
    monkeypatch.setenv("PYTHONHOME", "/malicious/test/home")

    observed = _run_json(
        "import json, os\n"
        "print(json.dumps({\n"
        '    "pythonpath": os.environ.get("PYTHONPATH"),\n'
        '    "pythonhome": os.environ.get("PYTHONHOME"),\n'
        "}))"
    )

    assert observed == {"pythonpath": None, "pythonhome": None}
    assert os.environ["PYTHONPATH"] == "/malicious/test/path"
    assert os.environ["PYTHONHOME"] == "/malicious/test/home"


def test_path_is_inherited_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_path = os.pathsep.join(("/agentipc/test/bin-one", "/agentipc/test/bin-two"))
    monkeypatch.setenv("PATH", parent_path)

    result = PythonSandbox().run(
        'import os\nprint(os.environ.get("PATH"))',
        timeout_sec=2.0,
    )

    assert result.exit_code == 0
    assert result.stdout == f"{parent_path}\n"
    assert os.environ["PATH"] == parent_path


def test_workspace_owns_cwd_home_and_temp_paths_and_is_cleaned_up() -> None:
    parent_cwd = Path.cwd()
    observed = _run_json(
        "import json, os, pathlib, tempfile\n"
        "print(json.dumps({\n"
        '    "cwd": os.getcwd(),\n'
        '    "home": os.environ["HOME"],\n'
        '    "userprofile": os.environ["USERPROFILE"],\n'
        '    "tmpdir": os.environ["TMPDIR"],\n'
        '    "tmp": os.environ["TMP"],\n'
        '    "temp": os.environ["TEMP"],\n'
        '    "path_home": str(pathlib.Path.home()),\n'
        '    "tempfile_tmpdir": tempfile.gettempdir(),\n'
        "}))"
    )

    workspace = Path(str(observed["cwd"]))
    expected = workspace.resolve()
    for key in (
        "home",
        "userprofile",
        "tmpdir",
        "tmp",
        "temp",
        "path_home",
        "tempfile_tmpdir",
    ):
        assert Path(str(observed[key])).resolve() == expected

    assert Path.cwd() == parent_cwd
    assert workspace.exists() is False


def test_parent_environment_is_unchanged_after_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HOME", "/parent/home/sentinel")
    monkeypatch.setenv("PATH", "/parent/path/sentinel")
    monkeypatch.setenv("AGENTIPC_TEST_SECRET", "still-parent-only")
    before = dict(os.environ)

    result = PythonSandbox().run('print("ok")', timeout_sec=2.0)

    assert result.exit_code == 0
    assert dict(os.environ) == before
    assert os.environ["HOME"] == "/parent/home/sentinel"
    assert os.environ["PATH"] == "/parent/path/sentinel"
    assert os.environ["AGENTIPC_TEST_SECRET"] == "still-parent-only"


def test_standard_library_imports_work_with_restricted_environment() -> None:
    result = PythonSandbox().run(
        "import json\n"
        "import pathlib\n"
        "import tempfile\n"
        "import os\n"
        "import sys\n"
        'print("stdlib-ok")\n',
        timeout_sec=2.0,
    )

    assert result.exit_code == 0
    assert result.stdout == "stdlib-ok\n"
    assert result.stderr == ""
    assert result.timed_out is False


def test_linux_resource_limits_are_applied_through_python_sandbox() -> None:
    if not limits.resource_limits_supported():
        pytest.skip("resource limits are not supported on this platform")

    observed = _run_json(
        "import json, resource\n"
        "print(json.dumps({\n"
        '    "cpu": resource.getrlimit(resource.RLIMIT_CPU),\n'
        '    "as": resource.getrlimit(resource.RLIMIT_AS),\n'
        "}))"
    )

    cpu_soft, cpu_hard = observed["cpu"]  # type: ignore[misc]
    as_soft, as_hard = observed["as"]  # type: ignore[misc]

    assert 0 < cpu_soft <= cpu_hard <= 5
    assert 0 < as_soft <= as_hard <= 2 * 1024 * 1024 * 1024


def test_unsupported_resource_limit_fallback_does_not_break_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def unsupported_helper() -> None:
        nonlocal calls
        calls += 1
        return None

    monkeypatch.setattr(
        python_runner,
        "build_resource_limit_preexec_fn",
        unsupported_helper,
    )

    result = python_runner.PythonSandbox().run('print("fallback-ok")', timeout_sec=2.0)

    assert calls == 1
    assert result.exit_code == 0
    assert result.stdout == "fallback-ok\n"
    assert result.stderr == ""
    assert result.timed_out is False
