from __future__ import annotations
def render_report(summary: dict) -> str:
    cols = ["Experiment","Mean latency (ms)","Mean tokens","Mean protocol bytes","Mean state bytes","Mean tool calls","Memory effective rate"]
    lines = ["# E1 A/B/C/D Ablation Experiment", "", "| " + " | ".join(cols) + " |", "|" + "|".join(["---"]*len(cols)) + "|"]
    for name in ("A","B","C","D"):
        s=summary[name]; lines.append(f"| {name} | {s['mean_latency_ms']:.3f} | {s['mean_tokens']:.2f} | {s['mean_protocol_bytes']:.2f} | {s['mean_state_bytes']:.2f} | {s['mean_tool_calls']:.2f} | {s['memory_effective_rate']:.3f} |")
    return "\n".join(lines) + "\n"
