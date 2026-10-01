from __future__ import annotations
from collections import defaultdict
from typing import Any

FIELDS = ("experiment","repeat","latency_ms","llm_prompt_tokens","llm_completion_tokens","llm_total_tokens","protocol_bytes","state_bytes","memory_used","memory_effective","memory_harmful","tool_call_count")

def flatten_record(record: Any, repeat: int = 0) -> dict[str, Any]:
    exp = getattr(record, "experiment", None)
    name = getattr(exp, "name", exp)
    name = getattr(name, "value", name)
    result = getattr(record, "run_result", record)
    metrics = getattr(result, "metrics", {}) or {}
    if hasattr(metrics, "model_dump"): metrics = metrics.model_dump()
    out = {"experiment": str(name), "repeat": repeat}
    for key in FIELDS[2:]: out[key] = metrics.get(key, 0)
    out["memory_harmful"] = bool(out["memory_harmful"])
    return out

def aggregate_records(records: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped = defaultdict(list)
    for row in records:
        if row.get("experiment") not in {"A","B","C","D"}: raise ValueError("invalid experiment")
        grouped[row["experiment"]].append(row)
    summary = {}
    for name in ("A","B","C","D"):
        rows = grouped.get(name, [])
        if not rows: raise ValueError(f"missing experiment {name}")
        n = len(rows)
        mean = lambda k: sum(float(r.get(k, 0)) for r in rows) / n
        summary[name] = {"mean_latency_ms": mean("latency_ms"), "mean_tokens": mean("llm_total_tokens"), "mean_protocol_bytes": mean("protocol_bytes"), "mean_state_bytes": mean("state_bytes"), "mean_tool_calls": mean("tool_call_count"), "memory_effective_rate": sum(bool(r.get("memory_effective")) for r in rows) / n}
    return summary
