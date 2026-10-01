from __future__ import annotations

import pytest

from agentipc.experiments.formal.e8_full_system import (
    _aggregate_e8,
    _verify_task_integrity,
)
from agentipc.experiments.formal.report import render_report


def _metrics(*, full: bool, repeat_phase: bool) -> dict[str, int | float | bool]:
    if full and repeat_phase:
        return {
            "message_count": 0,
            "text_chars": 0,
            "text_tokens": 0,
            "wire_chars": 0,
            "wire_tokens": 0,
            "wire_bytes": 0,
            "protocol_bytes": 0,
            "state_transfer_count": 0,
            "state_bytes": 0,
            "artifact_ref_count": 0,
            "artifact_payload_bytes": 0,
            "memory_retrieved": 1,
            "memory_used": 1,
            "memory_effective": 1,
            "memory_harmful": 0,
            "fast_path_hit_count": 1,
            "tool_call_count": 0,
            "repeated_tool_call_count": 0,
            "llm_call_count": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            "llm_usage_missing_count": 0,
            "llm_latency_ms": 0.0,
            "latency_ms": 10.0,
            "success": True,
        }
    return {
        "message_count": 8,
        "text_chars": 1000 if not full else 0,
        "text_tokens": 200 if not full else 0,
        "wire_chars": 1000 if not full else 500,
        "wire_tokens": 200 if not full else 100,
        "wire_bytes": 2000 if not full else 800,
        "protocol_bytes": 0 if not full else 800,
        "state_transfer_count": 0 if not full else 1,
        "state_bytes": 0 if not full else 256,
        "artifact_ref_count": 1,
        "artifact_payload_bytes": 1024,
        "memory_retrieved": 0,
        "memory_used": 0,
        "memory_effective": 0,
        "memory_harmful": 0,
        "fast_path_hit_count": 0,
        "tool_call_count": 1,
        "repeated_tool_call_count": 0,
        "llm_call_count": 2,
        "llm_prompt_tokens": 100,
        "llm_completion_tokens": 50,
        "llm_total_tokens": 150,
        "llm_usage_missing_count": 0,
        "llm_latency_ms": 100.0,
        "latency_ms": 1000.0 if not full else 900.0,
        "success": True,
    }


def _healthy_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in ("knowledge", "codeact"):
        for source_round in range(1, 6):
            for phase, sequence in (
                ("new", source_round),
                ("repeat", source_round + 5),
            ):
                task_hash = f"{group}-{source_round}".encode().hex()
                task_hash = (task_hash * 64)[:64]
                for config in ("A-Text", "D-Full"):
                    full = config == "D-Full"
                    fast_hit = int(full and phase == "repeat")
                    rows.append(
                        {
                            "experiment": "E8",
                            "group": group,
                            "config": config,
                            "repeat": 1,
                            "sequence": sequence,
                            "phase": phase,
                            "source_round": source_round,
                            "task_hash": task_hash,
                            "runtime_success": True,
                            "evaluation_pass": True,
                            "error": None,
                            "validated_fast_path_effective": fast_hit,
                            "validated_fast_path_harmful": 0,
                            "metrics": _metrics(
                                full=full,
                                repeat_phase=(phase == "repeat"),
                            ),
                        }
                    )
    return rows


def test_e8_healthy_full_system_comparison_passes() -> None:
    rows = _healthy_rows()
    _verify_task_integrity(rows)

    summary = _aggregate_e8(
        rows,
        repeat=1,
        provider={
            "llm_model": "fake-model",
            "api_region": "singapore",
            "timeout_sec": 120.0,
            "max_retries": 0,
        },
        git_sha="a" * 40,
    )

    assert summary["passed"] is True
    assert summary["integrity_pass"] is True
    assert summary["comparison_valid"] is True
    assert summary["quality_gate_pass"] is True
    assert summary["fast_path_safety_pass"] is True
    assert summary["row_count"] == 40

    overall = summary["overall"]
    assert overall["configs"]["A-Text"]["evaluation_pass_rate"] == 1.0
    assert overall["configs"]["D-Full"]["evaluation_pass_rate"] == 1.0
    assert (
        overall["phases"]["new"]["configs"]["D-Full"][
            "total_fast_path_hits"
        ]
        == 0
    )
    assert (
        overall["phases"]["repeat"]["configs"]["D-Full"][
            "total_fast_path_hits"
        ]
        == 10
    )
    assert overall["a_vs_full"]["provider_total_token_saving_pct"] > 0
    assert overall["a_vs_full"]["wire_token_saving_pct"] > 0

    report = render_report(summary)
    assert "# E8 End-to-End Full-System Benchmark" in report
    assert "A-Text" in report
    assert "D-Full" in report
    assert "New-task half" in report
    assert "Exact-repeat half" in report
    assert "50% exact-repeat" in report


def test_e8_infrastructure_failure_invalidates_comparison() -> None:
    rows = _healthy_rows()
    failed = next(
        row
        for row in rows
        if row["group"] == "knowledge"
        and row["config"] == "A-Text"
        and row["phase"] == "new"
    )
    failed["runtime_success"] = False
    failed["evaluation_pass"] = False
    failed["error"] = {
        "type": "APITimeoutError",
        "message": "Request timed out.",
    }

    summary = _aggregate_e8(rows, repeat=1)

    assert summary["passed"] is False
    assert summary["infrastructure_valid"] is False
    assert summary["comparison_valid"] is False
    assert summary["infrastructure_failure_count"] == 1

    report = render_report(summary)
    assert "Infrastructure failures: **1**" in report
    assert "diagnostic only" in report


def test_e8_integrity_rejects_mismatched_task_hash() -> None:
    rows = _healthy_rows()
    target = next(
        row
        for row in rows
        if row["group"] == "codeact"
        and row["config"] == "D-Full"
        and row["sequence"] == 1
    )
    target["task_hash"] = "f" * 64

    with pytest.raises(RuntimeError, match="integrity violation"):
        _verify_task_integrity(rows)
