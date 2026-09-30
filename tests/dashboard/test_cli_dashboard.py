from pathlib import Path
import socket

import pytest
from fastapi.testclient import TestClient

import agentipc.cli as cli
from agentipc.cli import build_parser, main


def test_dashboard_parser_defaults() -> None:
    args = build_parser().parse_args(["dashboard"])

    assert args.command == "dashboard"
    assert args.host == "127.0.0.1"
    assert args.port == 8000
    assert args.results_dir == "results"


def test_dashboard_parser_custom_values(tmp_path: Path) -> None:
    results_dir = tmp_path / "agentipc-results"
    args = build_parser().parse_args(
        [
            "dashboard",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
            "--results-dir",
            str(results_dir),
        ]
    )

    assert args.command == "dashboard"
    assert args.host == "0.0.0.0"
    assert args.port == 8765
    assert args.results_dir == str(results_dir)


@pytest.mark.parametrize("value", ["0", "-1", "65536", "not-a-number"])
def test_dashboard_parser_rejects_invalid_port(value: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["dashboard", "--port", value])

    assert exc_info.value.code != 0


@pytest.mark.parametrize("flag", ["--host", "--results-dir"])
def test_dashboard_parser_rejects_empty_strings(flag: str) -> None:
    with pytest.raises(SystemExit) as exc_info:
        build_parser().parse_args(["dashboard", flag, ""])

    assert exc_info.value.code != 0


def test_dashboard_cli_launches_existing_app_without_binding_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import uvicorn

    def fail_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("dashboard CLI unit test must not access the network")

    captured: dict[str, object] = {}

    def fake_run(app: object, *, host: str, port: int) -> None:
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port

        client = TestClient(app)  # type: ignore[arg-type]
        assert client.get("/").status_code == 200
        assert client.get("/").json() == {
            "service": "AgentIPC Dashboard",
            "status": "ok",
        }
        assert client.get("/api/runs").json() == {"runs": []}
        assert client.get("/dashboard/").status_code == 200

    monkeypatch.setattr(socket.socket, "connect", fail_connect)
    monkeypatch.setattr(uvicorn, "run", fake_run)
    results_dir = tmp_path / "not-created-yet"

    exit_code = main(
        [
            "dashboard",
            "--host",
            "127.0.0.1",
            "--port",
            "8765",
            "--results-dir",
            str(results_dir),
        ]
    )

    captured_io = capsys.readouterr()
    assert exit_code == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8765
    assert "URL: http://127.0.0.1:8765/dashboard/" in captured_io.out
    assert f"Results: {results_dir}" in captured_io.out
    assert captured_io.err == ""
    assert not results_dir.exists()


def test_dashboard_missing_optional_dependency_is_clean_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_load() -> tuple[object, object]:
        raise RuntimeError(
            "dashboard dependencies are not installed; install agentipc[dashboard]"
        )

    monkeypatch.setattr(cli, "_load_dashboard_runtime", fail_load)

    exit_code = main(["dashboard"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert (
        "agentipc: error: dashboard dependencies are not installed; "
        "install agentipc[dashboard]"
    ) in captured.err
    assert "Traceback" not in captured.err
