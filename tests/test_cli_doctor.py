import json
from pathlib import Path
import socket

import pytest

import agentipc.doctor as doctor
from agentipc.cli import main
from agentipc.doctor import DoctorCheck, run_doctor


def test_doctor_report_succeeds_for_core_environment() -> None:
    report = run_doctor()

    assert report.ok is True
    required = {check.name: check for check in report.checks if check.required}
    for name in (
        "python",
        "filesystem",
        "sqlite",
        "shared_memory",
        "provider_mock",
        "provider_hash_embedding",
    ):
        assert required[name].status == "PASS"


def test_doctor_json_cli_emits_one_parseable_document(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["doctor", "--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    parsed = json.loads(captured.out)
    assert parsed["ok"] is True
    assert isinstance(parsed["checks"], list)
    assert captured.err == ""


def test_doctor_human_cli_shows_names_and_statuses(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["doctor"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "PASS" in captured.out
    assert "python" in captured.out
    assert "filesystem" in captured.out
    assert "sqlite" in captured.out
    assert "shared_memory" in captured.out


def test_missing_optional_dependencies_are_warn_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(doctor.importlib_util, "find_spec", lambda name: None)

    report = run_doctor()

    assert report.ok is True
    optional = [check for check in report.checks if not check.required]
    assert optional
    assert all(check.status == "WARN" for check in optional)


def test_required_check_failure_makes_report_and_cli_fail(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        doctor,
        "_check_sqlite",
        lambda: DoctorCheck(
            name="sqlite",
            status="FAIL",
            required=True,
            message="forced test failure",
        ),
    )

    report = run_doctor()
    assert report.ok is False

    exit_code = main(["doctor", "--json"])
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert exit_code != 0
    assert parsed["ok"] is False


def test_doctor_is_offline_and_does_not_print_secrets(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secrets = {
        "OPENAI_API_KEY": "doctor-secret-openai",
        "AGENTIPC_LLM_API_KEY": "doctor-secret-agentipc",
    }
    for key, value in secrets.items():
        monkeypatch.setenv(key, value)

    def fail_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("doctor must not access the network")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)

    exit_code = main(["doctor", "--json"])
    captured = capsys.readouterr()
    assert exit_code == 0
    json.loads(captured.out)
    for secret in secrets.values():
        assert secret not in captured.out


def test_doctor_leaves_no_temporary_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    report = run_doctor()

    assert report.ok is True
    leftovers = [
        path.name
        for path in tmp_path.iterdir()
        if path.name.startswith(".agentipc-doctor-") or path.name == "doctor.tmp"
    ]
    assert leftovers == []
