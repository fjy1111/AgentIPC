"""FastAPI application factory for the AgentIPC Dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException

from agentipc.dashboard.results import load_run_detail, scan_results


def create_app(results_dir: str | Path = "results") -> FastAPI:
    """Create a Dashboard API application without scanning at import time."""

    if not isinstance(results_dir, (str, Path)):
        raise TypeError("results_dir must be a str or Path")

    results_root = Path(results_dir)
    app = FastAPI(title="AgentIPC Dashboard")

    @app.get("/")
    def root() -> dict[str, str]:
        return {"service": "AgentIPC Dashboard", "status": "ok"}

    @app.get("/api/runs")
    def list_runs() -> dict[str, list[dict[str, object]]]:
        try:
            runs = scan_results(results_root)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="results repository unavailable") from exc
        return {"runs": [run.model_dump(mode="json") for run in runs]}

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, object]:
        try:
            detail = load_run_detail(results_root, run_id)
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="results repository unavailable") from exc

        if detail is None:
            raise HTTPException(status_code=404, detail="run not found")

        summary_payload = detail.summary.model_dump(mode="json")
        derived = summary_payload.pop("derived")
        return {
            "run_id": detail.run_id,
            "summary": summary_payload,
            "derived": derived,
            "environment": detail.environment.model_dump(mode="json"),
            "report": {
                "title": detail.report_title,
                "size_bytes": detail.report_bytes,
            },
        }

    return app
