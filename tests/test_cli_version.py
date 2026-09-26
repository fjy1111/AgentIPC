import socket

import pytest

import agentipc
from agentipc.cli import main


def test_version_command_succeeds_and_prints_package_version(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert agentipc.__version__ in captured.out


def test_version_command_does_not_require_network_or_api_key(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    def fail_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("version command must not access the network")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)

    exit_code = main(["version"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert agentipc.__version__ in captured.out