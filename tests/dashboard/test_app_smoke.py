"""Smoke tests for the Dashboard FastAPI app factory."""

from pathlib import Path

from fastapi.testclient import TestClient

from agentipc.dashboard.app import create_app


def test_root_endpoint_is_json_health_response(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "missing-results"))

    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"service": "AgentIPC Dashboard", "status": "ok"}


def test_missing_results_directory_is_normal_empty_state(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "missing-results"))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json() == {"runs": []}


def test_repository_file_error_is_generic_500(tmp_path: Path) -> None:
    results_file = tmp_path / "results"
    results_file.write_text("not a directory", encoding="utf-8")
    client = TestClient(create_app(results_file))

    assert client.get("/").status_code == 200
    response = client.get("/api/runs")

    assert response.status_code == 500
    assert response.json() == {"detail": "results repository unavailable"}
    assert str(tmp_path) not in response.text
