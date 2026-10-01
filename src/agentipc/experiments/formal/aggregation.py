from __future__ import annotations

from collections import defaultdict
from typing import Any


FIELDS = (
    "experiment",
    "repeat",
    "latency_ms",
    "message_count",
    "text_chars",
    "text_tokens",
    "wire_chars",
    "wire_tokens",
    "wire_bytes",
    "protocol_bytes",
    "state_transfer_count",
    "state_bytes",
    "artifact_ref_count",
    "artifact_payload_bytes",
    "memory_retrieved",
    "memory_used",
    "memory_effective",
    "memory_harmful",
    "fast_path_hit_count",
    "tool_call_count",
    "repeated_tool_call_count",
    "llm_call_count",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_total_tokens",
    "llm_usage_missing_count",
    "llm_latency_ms",
    "success",
)


def flatten_record(record: Any, repeat: int = 0) -> dict[str, Any]:
    exp = getattr(record, "experiment", None)
    name = getattr(exp, "name", exp)
    name = getattr(name, "value", name)
    result = getattr(record, "run_result", record)
    metrics = getattr(result, "metrics", {}) or {}
    if hasattr(metrics, "model_dump"):
        metrics = metrics.model_dump()
    out = {"experiment": str(name), "repeat": repeat}
    for key in FIELDS[2:]:
        out[key] = metrics.get(key, False if key == "success" else 0)
    out["memory_harmful"] = int(out["memory_harmful"])
    out["success"] = bool(out["success"])
    return out


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if row.get("experiment") not in {"A", "B", "C", "D"}:
            raise ValueError("invalid experiment")
        grouped[row["experiment"]].append(row)

    summary: dict[str, dict[str, float]] = {}
    for name in ("A", "B", "C", "D"):
        rows = grouped.get(name, [])
        if not rows:
            raise ValueError(f"missing experiment {name}")
        n = len(rows)

        def mean(key: str) -> float:
            return sum(float(row.get(key, 0)) for row in rows) / n

        summary[name] = {
            "mean_latency_ms": mean("latency_ms"),
            "mean_message_count": mean("message_count"),
            "mean_text_chars": mean("text_chars"),
            "mean_text_tokens": mean("text_tokens"),
            "mean_wire_chars": mean("wire_chars"),
            "mean_wire_tokens": mean("wire_tokens"),
            "mean_wire_bytes": mean("wire_bytes"),
            "mean_protocol_bytes": mean("protocol_bytes"),
            "mean_state_transfer_count": mean("state_transfer_count"),
            "mean_state_bytes": mean("state_bytes"),
            "mean_artifact_ref_count": mean("artifact_ref_count"),
            "mean_artifact_payload_bytes": mean("artifact_payload_bytes"),
            "mean_tool_calls": mean("tool_call_count"),
            "mean_repeated_tool_calls": mean("repeated_tool_call_count"),
            "mean_llm_calls": mean("llm_call_count"),
            "mean_llm_prompt_tokens": mean("llm_prompt_tokens"),
            "mean_llm_completion_tokens": mean("llm_completion_tokens"),
            "mean_tokens": mean("llm_total_tokens"),
            "mean_llm_latency_ms": mean("llm_latency_ms"),
            "mean_memory_retrieved": mean("memory_retrieved"),
            "mean_memory_used": mean("memory_used"),
            "memory_effective_rate": sum(bool(row.get("memory_effective")) for row in rows) / n,
            "memory_harmful_rate": sum(bool(row.get("memory_harmful")) for row in rows) / n,
            "success_rate": sum(bool(row.get("success")) for row in rows) / n,
        }
    return summary
