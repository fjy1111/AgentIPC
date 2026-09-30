from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys

import pytest

import agentipc
import agentipc.dashboard.app as dashboard_app


def _installed_distribution() -> metadata.Distribution:
    return metadata.distribution("agentipc")


def test_distribution_identity_matches_package_version() -> None:
    installed = _installed_distribution()

    assert installed.metadata["Name"] == "agentipc"
    assert installed.version == agentipc.__version__
    assert installed.version == "0.1.0"


def test_distribution_declares_supported_python_and_console_script() -> None:
    installed = _installed_distribution()

    requires_python = installed.metadata.get("Requires-Python")
    assert requires_python is not None
    assert ">=3.10" in requires_python.replace(" ", "")

    entry_points = [
        entry
        for entry in installed.entry_points
        if entry.group == "console_scripts" and entry.name == "agentipc"
    ]
    assert len(entry_points) == 1
    assert entry_points[0].value == "agentipc.cli:main"


def test_dashboard_extra_declares_runtime_dependencies() -> None:
    installed = _installed_distribution()

    extras = {
        value.lower()
        for value in (installed.metadata.get_all("Provides-Extra") or [])
    }
    assert "dashboard" in extras

    requirements = [requirement.lower() for requirement in (installed.requires or [])]
    for package in ("fastapi", "uvicorn", "httpx"):
        assert any(
            requirement.startswith(package) and "dashboard" in requirement
            for requirement in requirements
        )


def test_default_static_dir_falls_back_to_installed_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = tmp_path / "site-packages" / "agentipc" / "dashboard" / "app.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("# synthetic module path\n", encoding="utf-8")

    prefix = tmp_path / "wheel-prefix"
    installed_static = prefix / "share" / "agentipc" / "dashboard"
    installed_static.mkdir(parents=True)

    monkeypatch.setattr(dashboard_app, "__file__", str(fake_module))
    monkeypatch.setattr(sys, "prefix", str(prefix))

    assert dashboard_app._resolve_static_dir(None) == installed_static


def test_explicit_missing_static_dir_does_not_fall_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = tmp_path / "site-packages" / "agentipc" / "dashboard" / "app.py"
    fake_module.parent.mkdir(parents=True)
    fake_module.write_text("# synthetic module path\n", encoding="utf-8")

    prefix = tmp_path / "wheel-prefix"
    installed_static = prefix / "share" / "agentipc" / "dashboard"
    installed_static.mkdir(parents=True)

    monkeypatch.setattr(dashboard_app, "__file__", str(fake_module))
    monkeypatch.setattr(sys, "prefix", str(prefix))

    explicit_missing = tmp_path / "explicit-missing-dashboard"
    assert dashboard_app._resolve_static_dir(explicit_missing) is None
