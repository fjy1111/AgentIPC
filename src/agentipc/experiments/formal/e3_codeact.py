from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from agentipc.config import AgentIPCConfig
from agentipc.experiments.real_bailian.config import load_real_bailian_config
from agentipc.experiments.real_bailian.runner import (
    build_recording_provider_bundle,
    run_codeact_mini_calibration,
)
from agentipc.providers.factory import ProviderBundle


def run_e3(
    *,
    root: Path,
    result_dir: Path,
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run independent CodeAct Round-2 -> Round-8 memory-reuse chains."""
    _validate_repeat(repeat)

    secret = load_real_bailian_config()
    llm_recorder, embedding_recorder = build_recording_provider_bundle(secret)
    provider_bundle = ProviderBundle(
        llm=llm_recorder,
        embedding=embedding_recorder,
    )

    rows: list[dict[str, Any]] = []
    repetitions: list[dict[str, Any]] = []
    for repeat_index in range(repeat):
        repeat_number = repeat_index + 1
        repeat_dir = result_dir / f"repeat-{repeat_number:03d}"
        summary, raw = run_codeact_mini_calibration(
            repo_root=root,
            result_dir=repeat_dir,
            config=AgentIPCConfig(
                llm_provider="openai",
                embedding_provider="openai",
                random_seed=42 + repeat_index,
            ),
            provider_bundle=provider_bundle,
            secret_config=secret,
        )
        repetitions.append(summary)
        rows.extend({**row, "repeat": repeat_number} for row in raw)

    return rows, _aggregate_codeact(repetitions)


def _aggregate_codeact(repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    round2 = [_round_result(item, 2) for item in repetitions]
    round8 = [_round_result(item, 8) for item in repetitions]
    pass_count = sum(item.get("passed") is True for item in repetitions)
    r2 = _aggregate_round(round2)
    r8 = _aggregate_round(round8)

    return {
        "experiment": "E3",
        "repeat": len(repetitions),
        "passed": pass_count == len(repetitions),
        "pass_count": pass_count,
        "pass_rate": pass_count / len(repetitions),
        "round2": r2,
        "round8": r8,
        "delta": {
            "latency_reduction_pct": _reduction_pct(
                r2["mean_latency_ms"],
                r8["mean_latency_ms"],
            ),
            "token_reduction_pct": _reduction_pct(
                r2["mean_tokens"],
                r8["mean_tokens"],
            ),
            "tool_call_reduction_pct": _reduction_pct(
                r2["mean_tool_calls"],
                r8["mean_tool_calls"],
            ),
        },
    }


def _aggregate_round(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "evaluation_pass_rate": _rate(
            rows,
            lambda row: row.get("evaluation_pass") is True,
        ),
        "identity_operation_rate": _rate(
            rows,
            lambda row: row.get("execution_operation") == "identity",
        ),
        "mean_latency_ms": _mean_metric(rows, "latency_ms"),
        "mean_tokens": _mean_metric(rows, "llm_total_tokens"),
        "mean_protocol_bytes": _mean_metric(rows, "protocol_bytes"),
        "mean_state_bytes": _mean_metric(rows, "state_bytes"),
        "mean_tool_calls": _mean_metric(rows, "tool_call_count"),
        "mean_repeated_tool_calls": _mean_metric(rows, "repeated_tool_call_count"),
        "mean_memory_retrieved": _mean_metric(rows, "memory_retrieved"),
        "mean_memory_used": _mean_metric(rows, "memory_used"),
        "strict_memory_effective_rate": _rate(
            rows,
            lambda row: row["memory_validation"]["runtime_memory_effective_strict"] > 0,
        ),
        "validated_memory_effective_rate": _rate(
            rows,
            lambda row: row["memory_validation"]["validated_memory_effective"] > 0,
        ),
    }


def _round_result(summary: dict[str, Any], round_number: int) -> dict[str, Any]:
    for row in summary.get("rounds", []):
        if row.get("round") == round_number:
            return row
    raise ValueError(f"CodeAct repetition missing round {round_number}")


def _mean_metric(rows: list[dict[str, Any]], name: str) -> float:
    return sum(float(row["metrics"].get(name, 0)) for row in rows) / len(rows)


def _rate(
    rows: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> float:
    return sum(bool(predicate(row)) for row in rows) / len(rows)


def _reduction_pct(baseline: float, reused: float) -> float:
    if baseline <= 0.0:
        return 0.0
    return (baseline - reused) / baseline * 100.0


def _validate_repeat(repeat: int) -> None:
    if type(repeat) is not int:
        raise TypeError("repeat must be an int")
    if repeat < 1:
        raise ValueError("repeat must be >= 1")
