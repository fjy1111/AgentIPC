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
    if summary.get("experiment") == "E7":
        return _render_e7(summary)
    if summary.get("experiment") == "E8":
        return _render_e8(summary)
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
    provider = summary.get("provider", {})
    timeout_sec = provider.get("timeout_sec")
    max_retries = provider.get("max_retries")
    provider_model = provider.get("llm_model")
    provider_region = provider.get("api_region")
    git_sha = summary.get("git_sha")

    lines = [
        "# E6 Real Provider Memory Fast Path",
        "",
        f"Overall pass: **{summary['passed']}**",
        f"Infrastructure valid: **{summary.get('infrastructure_valid', True)}**",
        f"Comparison valid: **{summary.get('comparison_valid', True)}**",
        f"Fast-path safety pass: **{summary.get('fast_path_safety_pass', summary['passed'])}**",
        f"Infrastructure failures: **{summary.get('infrastructure_failure_count', 0)}**",
        f"Workload: {summary['workload']}",
        f"Match: {summary['match_method']}",
    ]
    if provider_model:
        lines.append(f"LLM model: `{provider_model}`")
    if provider_region:
        lines.append(f"API region: `{provider_region}`")
    if timeout_sec is not None:
        lines.append(f"Provider timeout: **{float(timeout_sec):.0f}s**")
    if max_retries is not None:
        lines.append(f"Provider max retries: **{int(max_retries)}**")
    if git_sha:
        lines.append(f"Git SHA: `{git_sha}`")
    if summary.get("infrastructure_failure_types"):
        lines.append(
            "Infrastructure failure types: "
            + ", ".join(f"`{item}`" for item in summary["infrastructure_failure_types"])
        )
    lines.append("")

    for group in ("knowledge", "codeact"):
        data = summary["groups"][group]
        lines.extend([
            f"## {group.title()}",
            "",
            f"Comparison valid: **{data.get('comparison_valid', True)}**",
            "",
            "| Config | Eval pass (all) | Eval pass (non-infra) | Infra failures | Provider tokens | LLM calls | Wire tokens | Tool calls | Mean latency ms | Fast hits | Strict memory mismatch | Validated fast-path harmful |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for name in ("C", "D", "D-Fast"):
            item = data["configs"][name]
            lines.append(
                f"| {name} | {item['evaluation_pass_rate']:.1%} | "
                f"{item.get('evaluation_pass_rate_excluding_infrastructure', item['evaluation_pass_rate']):.1%} | "
                f"{item.get('infrastructure_failure_count', 0)} | "
                f"{item['total_llm_tokens']} | {item['total_llm_calls']} | "
                f"{item['total_wire_tokens']} | {item['total_tool_calls']} | "
                f"{item['mean_latency_ms']:.3f} | {item['total_fast_path_hits']} | "
                f"{item.get('strict_memory_mismatch_count', item.get('total_memory_harmful', 0))} | "
                f"{item.get('validated_fast_path_harmful_rate', item.get('wrong_harmful_memory_rate', 0)):.1%} |"
            )
        delta = data["c_vs_d_fast"]
        delta_heading = (
            "C vs D-Fast:"
            if data.get("comparison_valid", True)
            else "C vs D-Fast (diagnostic only; comparison invalid):"
        )
        lines.extend([
            "",
            delta_heading,
            f"- Provider prompt token saving: {delta['provider_prompt_token_saving_pct']:.2f}%",
            f"- Provider total token saving: {delta['provider_total_token_saving_pct']:.2f}%",
            f"- LLM call reduction: {delta['llm_call_reduction_pct']:.2f}%",
            f"- Wire token saving: {delta['wire_token_saving_pct']:.2f}%",
            f"- Latency reduction: {delta['latency_reduction_pct']:.2f}%",
            f"- 0% repeat control fast hits: {data['zero_repeat_control']['total_fast_path_hits']}",
            "",
        ])

    lines.extend(
        [
            "Strict memory mismatch means the regenerated answer string differed from "
            "the historical answer; it is not an evaluator failure.",
            "",
            "Validated fast-path harmful is evaluator-based and applies only to "
            "fast-path hits.",
        ]
    )
    return "\n".join(lines) + "\n"




def _render_e7(summary: dict[str, Any]) -> str:
    lines = [
        "# E7 Cross-Process Non-Text State Exchange",
        "",
        f"Overall pass: **{summary['passed']}**",
        f"Correctness pass: **{summary['correctness_pass']}**",
        f"Accounting pass: **{summary['accounting_pass']}**",
        f"Repeat: **{summary['repeat']}**",
        f"Token method: `{summary['token_method']}`",
        f"Process model: {summary['process_model']}",
        f"IPC transport: {summary['ipc_transport']}",
        f"Consumer operation: `{summary['consumer_operation']}`",
        "",
        "| State bytes | Vector dim | JSON wire bytes | SHM+Ref wire bytes | "
        "Wire byte saving | JSON wire tokens | SHM+Ref wire tokens | "
        "Wire token saving | JSON E2E ms | SHM E2E ms | E2E latency reduction | "
        "SHM state bytes | Correctness |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, payload in sorted(
        summary["payloads"].items(),
        key=lambda item: int(item[0]),
    ):
        json_mode = payload["modes"]["json_materialized"]
        shm_mode = payload["modes"]["shm_ref"]
        savings = payload["savings"]
        correctness = min(json_mode["success_rate"], shm_mode["success_rate"])
        lines.append(
            f"| {payload['state_bytes']} | {payload['vector_dim']} | "
            f"{json_mode['mean_wire_bytes']:.2f} | "
            f"{shm_mode['mean_wire_bytes']:.2f} | "
            f"{savings['wire_byte_saving_pct']:.2f}% | "
            f"{json_mode['mean_wire_tokens']:.2f} | "
            f"{shm_mode['mean_wire_tokens']:.2f} | "
            f"{savings['wire_token_saving_pct']:.2f}% | "
            f"{json_mode['mean_end_to_end_ms']:.3f} | "
            f"{shm_mode['mean_end_to_end_ms']:.3f} | "
            f"{savings['end_to_end_latency_reduction_pct']:.2f}% | "
            f"{shm_mode['mean_state_bytes']:.2f} | "
            f"{correctness:.1%} |"
        )

    lines.extend(
        [
            "",
            "## Measurement scope",
            "",
            f"- Timing: {summary['timing_scope']}",
            f"- Wire accounting: {summary['wire_scope']}",
            "",
            "AgentIPC does **not** claim zero-copy in E7. "
            "The current `StateHub.resolve_array()` returns an owned ndarray copy; "
            "the measured mechanism is binary non-text state exchange through "
            "Linux SharedMemory + StateRef, avoiding materialization of the numeric "
            "state into JSON on the Agent-to-Agent control path.",
            "",
            summary["note"],
        ]
    )
    return "\n".join(lines) + "\n"

def _render_e8(summary: dict[str, Any]) -> str:
    provider = summary.get("provider", {})
    overall = summary["overall"]

    lines = [
        "# E8 End-to-End Full-System Benchmark",
        "",
        f"Overall pass: **{summary['passed']}**",
        f"Integrity pass: **{summary['integrity_pass']}**",
        f"Infrastructure valid: **{summary['infrastructure_valid']}**",
        f"Comparison valid: **{summary['comparison_valid']}**",
        f"Quality gate pass: **{summary['quality_gate_pass']}**",
        f"Fast-path safety pass: **{summary['fast_path_safety_pass']}**",
        f"Infrastructure failures: **{summary['infrastructure_failure_count']}**",
        f"Rows: **{summary['row_count']}/{summary['expected_row_count']}**",
        f"Workload: {summary['workload']} (50% exact-repeat)",
        f"Baseline: {summary['baseline']}",
        f"Full system: {summary['full_system']}",
    ]
    if provider.get("llm_model"):
        lines.append(f"LLM model: `{provider['llm_model']}`")
    if provider.get("embedding_model"):
        lines.append(
            f"Embedding: `{provider['embedding_model']}` "
            f"(dim={provider.get('embedding_dim')})"
        )
    if provider.get("api_region"):
        lines.append(f"API region: `{provider['api_region']}`")
    if provider.get("timeout_sec") is not None:
        lines.append(
            f"Provider timeout: **{float(provider['timeout_sec']):.0f}s**"
        )
    if provider.get("max_retries") is not None:
        lines.append(
            f"Provider max retries: **{int(provider['max_retries'])}**"
        )
    if summary.get("git_sha"):
        lines.append(f"Git SHA: `{summary['git_sha']}`")
    if summary.get("infrastructure_failure_types"):
        lines.append(
            "Infrastructure failure types: "
            + ", ".join(
                f"`{name}`"
                for name in summary["infrastructure_failure_types"]
            )
        )

    lines.extend(
        [
            "",
            "## Overall",
            "",
            "| Config | Eval pass | Provider tokens | LLM calls | "
            "Wire tokens | Wire bytes | Messages | Tool calls | "
            "State transfers | State bytes | Fast hits | Mean latency ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("A-Text", "D-Full"):
        item = overall["configs"][name]
        lines.append(
            f"| {name} | {item['evaluation_pass_rate']:.1%} | "
            f"{item['total_llm_tokens']} | {item['total_llm_calls']} | "
            f"{item['total_wire_tokens']} | {item['total_wire_bytes']} | "
            f"{item['total_message_count']} | {item['total_tool_calls']} | "
            f"{item['total_state_transfer_count']} | {item['total_state_bytes']} | "
            f"{item['total_fast_path_hits']} | {item['mean_latency_ms']:.3f} |"
        )

    _append_e8_delta(
        lines,
        "A-Text vs D-Full — overall",
        overall["a_vs_full"],
        valid=overall["comparison_valid"],
    )

    for phase, title in (
        ("new", "New-task half"),
        ("repeat", "Exact-repeat half"),
    ):
        phase_data = overall["phases"][phase]
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Config | Tasks | Eval pass | Provider tokens | LLM calls | "
                "Wire tokens | Wire bytes | Tool calls | Fast hits | "
                "Mean latency ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("A-Text", "D-Full"):
            item = phase_data["configs"][name]
            lines.append(
                f"| {name} | {item['task_count']} | "
                f"{item['evaluation_pass_rate']:.1%} | "
                f"{item['total_llm_tokens']} | {item['total_llm_calls']} | "
                f"{item['total_wire_tokens']} | {item['total_wire_bytes']} | "
                f"{item['total_tool_calls']} | {item['total_fast_path_hits']} | "
                f"{item['mean_latency_ms']:.3f} |"
            )
        _append_e8_delta(
            lines,
            f"A-Text vs D-Full — {title.lower()}",
            phase_data["a_vs_full"],
            valid=phase_data["comparison_valid"],
        )

    for group in ("knowledge", "codeact"):
        data = summary["groups"][group]
        lines.extend(
            [
                "",
                f"## {group.title()}",
                "",
                "| Config | Eval pass | Provider tokens | LLM calls | "
                "Wire tokens | Wire bytes | Tool calls | Fast hits | "
                "Mean latency ms |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for name in ("A-Text", "D-Full"):
            item = data["configs"][name]
            lines.append(
                f"| {name} | {item['evaluation_pass_rate']:.1%} | "
                f"{item['total_llm_tokens']} | {item['total_llm_calls']} | "
                f"{item['total_wire_tokens']} | {item['total_wire_bytes']} | "
                f"{item['total_tool_calls']} | {item['total_fast_path_hits']} | "
                f"{item['mean_latency_ms']:.3f} |"
            )
        _append_e8_delta(
            lines,
            f"A-Text vs D-Full — {group}",
            data["a_vs_full"],
            valid=data["comparison_valid"],
        )

    lines.extend(
        [
            "",
            "## Measurement scope",
            "",
            f"- Fairness: {summary['fairness']}",
            f"- Accounting: {summary['measurement_note']}",
            "- The overall result intentionally uses a 50% exact-repeat workload. "
            "Use the New-task half to discuss fresh-task full-stack cost/benefit; "
            "use the Exact-repeat half to discuss validated memory reuse.",
        ]
    )
    return "\n".join(lines) + "\n"


def _append_e8_delta(
    lines: list[str],
    title: str,
    delta: dict[str, float],
    *,
    valid: bool,
) -> None:
    heading = (
        f"### {title}"
        if valid
        else f"### {title} (diagnostic only; comparison invalid)"
    )
    lines.extend(
        [
            "",
            heading,
            "",
            f"- Provider prompt token saving: "
            f"{delta['provider_prompt_token_saving_pct']:.2f}%",
            f"- Provider total token saving: "
            f"{delta['provider_total_token_saving_pct']:.2f}%",
            f"- LLM call reduction: "
            f"{delta['llm_call_reduction_pct']:.2f}%",
            f"- Message reduction: "
            f"{delta['message_reduction_pct']:.2f}%",
            f"- Wire token saving: "
            f"{delta['wire_token_saving_pct']:.2f}%",
            f"- Wire byte saving: "
            f"{delta['wire_byte_saving_pct']:.2f}%",
            f"- Tool-call reduction: "
            f"{delta['tool_call_reduction_pct']:.2f}%",
            f"- Latency reduction: "
            f"{delta['latency_reduction_pct']:.2f}%",
        ]
    )

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
