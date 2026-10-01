from __future__ import annotations

from typing import Any


def render_report(summary: dict[str, Any]) -> str:
    if all(name in summary for name in ("A", "B", "C", "D")):
        return _render_e1(summary)
    if summary.get("experiment") in {"E2", "E3"}:
        return _render_continuous_task(summary)
    if summary.get("experiment") == "E5":
        return _render_e5(summary)
    if summary.get("experiment") == "E6":
        return _render_e6(summary)
    if "inproc" in summary and "shm" in summary:
        return _render_e4(summary)
    raise ValueError("unsupported formal experiment summary")


def _render_e1(summary: dict[str, Any]) -> str:
    cols = [
        "Experiment",
        "Mean latency (ms)",
        "Mean tokens",
        "Mean protocol bytes",
        "Mean state bytes",
        "Mean tool calls",
        "Memory effective rate",
    ]
    lines = [
        "# E1 A/B/C/D Ablation Experiment",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for name in ("A", "B", "C", "D"):
        item = summary[name]
        lines.append(
            f"| {name} | {item.get('mean_latency_ms', 0):.3f} | "
            f"{item.get('mean_tokens', 0):.2f} | "
            f"{item.get('mean_protocol_bytes', 0):.2f} | "
            f"{item.get('mean_state_bytes', 0):.2f} | "
            f"{item.get('mean_tool_calls', item.get('tool_call_count', 0)):.2f} | "
            f"{item.get('memory_effective_rate', item.get('memory_effective', 0)):.3f} |"
        )
    return "\n".join(lines) + "\n"


def _render_continuous_task(summary: dict[str, Any]) -> str:
    experiment = summary["experiment"]
    title = "Knowledge Memory Reuse" if experiment == "E2" else "CodeAct Memory Reuse"
    cols = [
        "Round",
        "Eval pass rate",
        "Latency (ms)",
        "Tokens",
        "Tool calls",
        "Memory retrieved",
        "Memory used",
        "Validated effective rate",
    ]
    lines = [
        f"# {experiment} {title}",
        "",
        f"Overall pass: **{summary['pass_count']}/{summary['repeat']}** "
        f"({summary['pass_rate']:.1%})",
        "",
        "| " + " | ".join(cols) + " |",
        "|" + "|".join(["---"] * len(cols)) + "|",
    ]
    for key, label in (("round2", "Round 2"), ("round8", "Round 8")):
        item = summary[key]
        lines.append(
            f"| {label} | {item['evaluation_pass_rate']:.1%} | "
            f"{item['mean_latency_ms']:.3f} | {item['mean_tokens']:.2f} | "
            f"{item['mean_tool_calls']:.2f} | "
            f"{item['mean_memory_retrieved']:.2f} | "
            f"{item['mean_memory_used']:.2f} | "
            f"{item['validated_memory_effective_rate']:.1%} |"
        )

    delta = summary["delta"]
    lines.extend(
        [
            "",
            "## Reuse delta",
            "",
            f"- Latency reduction: {delta['latency_reduction_pct']:.2f}%",
            f"- Token reduction: {delta['token_reduction_pct']:.2f}%",
        ]
    )
    if "tool_call_reduction_pct" in delta:
        lines.append(f"- Tool-call reduction: {delta['tool_call_reduction_pct']:.2f}%")
    return "\n".join(lines) + "\n"


def _render_e5(summary: dict[str, Any]) -> str:
    lines = [
        "# E5 Communication Cost Benchmark",
        "",
        f"Tokenizer: `{summary['token_method']}`",
        "",
        "| Payload | Mode | Wire chars | Wire tokens | Wire bytes | Artifact payload bytes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for _, payload in sorted(summary["payloads"].items(), key=lambda item: int(item[0])):
        size = payload["payload_bytes"]
        for mode in ("A", "B", "C"):
            item = payload["modes"][mode]
            lines.append(
                f"| {size} | {mode} | {item['mean_wire_chars']:.2f} | "
                f"{item['mean_wire_tokens']:.2f} | {item['mean_wire_bytes']:.2f} | "
                f"{item['mean_artifact_payload_bytes']:.2f} |"
            )
        lines.append("")
        lines.append(
            f"- {size} B vs A: token {payload['savings_vs_A']['B']['wire_token_saving_pct']:.2f}%, "
            f"byte {payload['savings_vs_A']['B']['wire_byte_saving_pct']:.2f}%"
        )
        lines.append(
            f"- {size} C vs A: token {payload['savings_vs_A']['C']['wire_token_saving_pct']:.2f}%, "
            f"byte {payload['savings_vs_A']['C']['wire_byte_saving_pct']:.2f}%, "
            f"total-transfer {payload['savings_vs_A']['C']['total_transfer_saving_pct']:.2f}%"
        )
        lines.append("")
    lines.append(summary["note"])
    return "\n".join(lines) + "\n"


def _render_e6(summary: dict[str, Any]) -> str:
    lines = [
        "# E6 Real Provider Memory Fast Path",
        "",
        f"Overall pass: **{summary['passed']}**",
        f"Workload: {summary['workload']}",
        f"Match: {summary['match_method']}",
        "",
    ]
    for group in ("knowledge", "codeact"):
        data = summary["groups"][group]
        lines.extend([
            f"## {group.title()}",
            "",
            "| Config | Eval pass | Provider tokens | LLM calls | Wire tokens | Tool calls | Mean latency ms | Fast hits | Harmful rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in ("C", "D", "D-Fast"):
            item = data["configs"][name]
            lines.append(
                f"| {name} | {item['evaluation_pass_rate']:.1%} | "
                f"{item['total_llm_tokens']} | {item['total_llm_calls']} | "
                f"{item['total_wire_tokens']} | {item['total_tool_calls']} | "
                f"{item['mean_latency_ms']:.3f} | {item['total_fast_path_hits']} | "
                f"{item['wrong_harmful_memory_rate']:.1%} |"
            )
        delta = data["c_vs_d_fast"]
        lines.extend([
            "",
            "C vs D-Fast:",
            f"- Provider prompt token saving: {delta['provider_prompt_token_saving_pct']:.2f}%",
            f"- Provider total token saving: {delta['provider_total_token_saving_pct']:.2f}%",
            f"- LLM call reduction: {delta['llm_call_reduction_pct']:.2f}%",
            f"- Wire token saving: {delta['wire_token_saving_pct']:.2f}%",
            f"- Latency reduction: {delta['latency_reduction_pct']:.2f}%",
            f"- 0% repeat control fast hits: {data['zero_repeat_control']['total_fast_path_hits']}",
            "",
        ])
    return "\n".join(lines) + "\n"


def _render_e4(summary: dict[str, Any]) -> str:
    lines = [
        "# E4 State Transport Comparison",
        "",
        "| Transport | Mean latency (ms) | Mean state bytes |",
        "|---|---:|---:|",
    ]
    for name in ("inproc", "shm"):
        item = summary[name]
        lines.append(
            f"| {name} | {item.get('mean_latency_ms', 0):.3f} | "
            f"{item.get('mean_state_bytes', 0):.2f} |"
        )
    return "\n".join(lines) + "\n"
