from __future__ import annotations
def render_report(summary: dict) -> str:
    cols = ["Experiment","Mean latency (ms)","Mean tokens","Mean protocol bytes","Mean state bytes","Mean tool calls","Memory effective rate"]
    lines = ["# E1 A/B/C/D Ablation Experiment", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for name in summary:
        s=summary[name]; lines.append(f"| {name} | {s.get('mean_latency_ms', 0):.3f} | {s.get('mean_tokens', 0):.2f} | {s.get('mean_protocol_bytes', 0):.2f} | {s.get('mean_state_bytes', 0):.2f} | {s.get('mean_tool_calls', s.get('tool_call_count', 0)):.2f} | {s.get('memory_effective_rate', s.get('memory_effective', 0)):.3f} |")
    return "\n".join(lines) + "\n"
