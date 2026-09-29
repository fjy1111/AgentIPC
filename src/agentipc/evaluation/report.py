"""Markdown report renderer for A/B/C/D benchmark summaries.

This module renders BenchmarkSummary into human-readable markdown reports
with tables for communication, state, memory, LLM usage, latency, and
derived comparisons, plus scientific interpretation notes.
"""

from __future__ import annotations

from pathlib import Path

from agentipc.evaluation.io import BenchmarkSummary


def render_markdown_report(
    summary: BenchmarkSummary,
) -> str:
    """Render BenchmarkSummary as a human-readable markdown report.

    This function produces a deterministic markdown document with:
    - Overview: total records, task/seed counts, success rates
    - Communication table: message count, text chars/tokens, protocol bytes, artifacts
    - State table: state transfers and bytes
    - Memory table: retrieved/used/effective/harmful counts, hit rates
    - LLM usage table: calls, prompt/completion/total tokens, usage missing
    - Latency table: total and LLM latency
    - Derived comparisons: token/char saving, latency improvement, repeat work reduction, effective hit rate
    - Interpretation notes: scientific boundaries for metrics

    Args:
        summary: BenchmarkSummary instance to render

    Returns:
        Markdown report string ending with newline

    Raises:
        TypeError: If summary is not a BenchmarkSummary
    """
    if not isinstance(summary, BenchmarkSummary):
        raise TypeError("summary must be a BenchmarkSummary")

    lines = []

    # Title
    lines.append("# AgentIPC Benchmark Report")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- Total records: {summary.total_records}")
    lines.append(f"- Task count: {summary.task_count}")
    lines.append(f"- Seed count: {summary.seed_count}")
    lines.append("")

    lines.append("### Success Rates")
    lines.append("")
    lines.append("| Experiment | Run Count | Success Count | Success Rate |")
    lines.append("|------------|-----------|---------------|--------------|")

    for exp_name in ["A", "B", "C", "D"]:
        exp = summary.experiments[exp_name]
        success_rate_pct = exp.success_rate * 100.0
        lines.append(
            f"| {exp_name} | {exp.run_count} | {exp.success_count} | {success_rate_pct:.2f}% |"
        )

    lines.append("")

    # Communication
    lines.append("## Communication")
    lines.append("")
    lines.append("| Experiment | Message Count | Text Chars | Text Tokens | Protocol Bytes | Artifact Refs |")
    lines.append("|------------|---------------|------------|-------------|----------------|---------------|")

    for exp_name in ["A", "B", "C", "D"]:
        exp = summary.experiments[exp_name]
        msg_count = exp.metrics["message_count"].mean
        text_chars = exp.metrics["text_chars"].mean
        text_tokens = exp.metrics["text_tokens"].mean
        protocol_bytes = exp.metrics["protocol_bytes"].mean
        artifact_refs = exp.metrics["artifact_ref_count"].mean

        lines.append(
            f"| {exp_name} | {msg_count:.1f} | {text_chars:.1f} | {text_tokens:.1f} | {protocol_bytes:.1f} | {artifact_refs:.1f} |"
        )

    lines.append("")

    # State
    lines.append("## State")
    lines.append("")
    lines.append("| Experiment | State Transfers | State Bytes |")
    lines.append("|------------|-----------------|-------------|")

    for exp_name in ["A", "B", "C", "D"]:
        exp = summary.experiments[exp_name]
        state_transfers = exp.metrics["state_transfer_count"].mean
        state_bytes = exp.metrics["state_bytes"].mean

        lines.append(
            f"| {exp_name} | {state_transfers:.1f} | {state_bytes:.1f} |"
        )

    lines.append("")

    # Memory
    lines.append("## Memory")
    lines.append("")
    lines.append("| Experiment | Memory Retrieved | Memory Used | Memory Effective | Memory Harmful | Effective Hit Rate |")
    lines.append("|------------|------------------|-------------|------------------|----------------|--------------------|")

    for exp_name in ["A", "B", "C", "D"]:
        exp = summary.experiments[exp_name]
        memory_retrieved = exp.metrics["memory_retrieved"].mean
        memory_used = exp.metrics["memory_used"].mean
        memory_effective = exp.metrics["memory_effective"].mean
        memory_harmful = exp.metrics["memory_harmful"].mean

        # Compute effective hit rate from derived metrics
        if exp_name == "A":
            hit_rate_str = "N/A"
        else:
            # Extract from derived: B from B_vs_A, C from C_vs_B, D from D_vs_C
            derived_key = {
                "B": "B_vs_A",
                "C": "C_vs_B",
                "D": "D_vs_C",
            }[exp_name]

            hit_rate = summary.derived[derived_key].effective_hit_rate
            if hit_rate is None:
                hit_rate_str = "N/A"
            else:
                hit_rate_str = f"{hit_rate * 100.0:.2f}%"

        lines.append(
            f"| {exp_name} | {memory_retrieved:.1f} | {memory_used:.1f} | {memory_effective:.1f} | {memory_harmful:.1f} | {hit_rate_str} |"
        )

    lines.append("")

    # LLM Usage
    lines.append("## LLM Usage")
    lines.append("")
    lines.append("| Experiment | LLM Calls | Prompt Tokens | Completion Tokens | Total Tokens | Usage Missing |")
    lines.append("|------------|-----------|---------------|-------------------|--------------|---------------|")

    for exp_name in ["A", "B", "C", "D"]:
        exp = summary.experiments[exp_name]
        llm_calls = exp.metrics["llm_call_count"].mean
        prompt_tokens = exp.metrics["llm_prompt_tokens"].mean
        completion_tokens = exp.metrics["llm_completion_tokens"].mean
        total_tokens = exp.metrics["llm_total_tokens"].mean
        usage_missing = exp.metrics["llm_usage_missing_count"].mean

        lines.append(
            f"| {exp_name} | {llm_calls:.1f} | {prompt_tokens:.1f} | {completion_tokens:.1f} | {total_tokens:.1f} | {usage_missing:.1f} |"
        )

    lines.append("")

    # Latency
    lines.append("## Latency")
    lines.append("")
    lines.append("| Experiment | Total Latency (ms) | LLM Latency (ms) |")
    lines.append("|------------|-----------------------|------------------|")

    for exp_name in ["A", "B", "C", "D"]:
        exp = summary.experiments[exp_name]
        total_latency = exp.metrics["latency_ms"].mean
        llm_latency = exp.metrics["llm_latency_ms"].mean

        lines.append(
            f"| {exp_name} | {total_latency:.1f} | {llm_latency:.1f} |"
        )

    lines.append("")

    # Derived Comparisons
    lines.append("## Derived Comparisons")
    lines.append("")
    lines.append("| Comparison | Text Token Saving | Text Char Saving | Latency Improvement | Repeat Work Reduction | Effective Hit Rate |")
    lines.append("|------------|-------------------|------------------|---------------------|----------------------|--------------------|")

    for comparison_name in ["B vs A", "C vs B", "D vs C"]:
        derived_key = comparison_name.replace(" ", "_")
        derived = summary.derived[derived_key]

        token_saving = _format_rate(derived.token_saving_rate)
        char_saving = _format_rate(derived.char_saving_rate)
        latency_improvement = _format_rate(derived.latency_improvement_rate)
        repeat_reduction = _format_rate(derived.repeat_work_reduction_rate)
        hit_rate = _format_rate(derived.effective_hit_rate)

        lines.append(
            f"| {comparison_name} | {token_saving} | {char_saving} | {latency_improvement} | {repeat_reduction} | {hit_rate} |"
        )

    lines.append("")

    # Interpretation Notes
    lines.append("## Interpretation Notes")
    lines.append("")
    lines.append("### Metric Definitions and Scientific Boundaries")
    lines.append("")

    lines.append("**text_tokens vs llm_total_tokens:**")
    lines.append("")
    lines.append(
        "`text_tokens` is the AgentIPC TextCounter's communication-side token estimate, "
        "computed from text characters using a heuristic or tiktoken (when available). "
        "This metric estimates the size of text messages passed between agents in the IPC layer."
    )
    lines.append("")
    lines.append(
        "`llm_prompt_tokens`, `llm_completion_tokens`, and `llm_total_tokens` are the "
        "actual token counts reported by the LLM provider (when the provider returns usage data). "
        "These represent model-billed tokens, not IPC communication tokens."
    )
    lines.append("")
    lines.append(
        "**Important:** Text token saving rates in derived comparisons reflect changes in "
        "IPC communication volume, not LLM billing savings. LLM token counts are reported "
        "separately in the LLM Usage section."
    )
    lines.append("")

    lines.append("**protocol_bytes vs text_chars:**")
    lines.append("")
    lines.append(
        "`protocol_bytes` measures the size of serialized protocol messages (JSON encoding). "
        "`text_chars` measures character count in text mode. These are different units and "
        "should not be directly compared as \"byte savings.\""
    )
    lines.append("")

    lines.append("**Derived Comparisons:**")
    lines.append("")
    lines.append(
        "The three derived comparisons (B vs A, C vs B, D vs C) are adjacent ablation steps:"
    )
    lines.append("")
    lines.append("- **B vs A:** Text → Structured protocol")
    lines.append("- **C vs B:** + State exchange")
    lines.append("- **D vs C:** + Shared memory")
    lines.append("")
    lines.append(
        "Each comparison isolates one infrastructure change. Positive rates indicate the "
        "candidate improved over the baseline (saved tokens, reduced latency, etc.). "
        "Negative rates indicate regression. N/A indicates undefined (zero denominator)."
    )
    lines.append("")

    # Final newline
    result = "\n".join(lines) + "\n"
    return result


def _format_rate(rate: float | None) -> str:
    """Format a rate as percentage string or N/A.

    Args:
        rate: Rate value (0.25 = 25%) or None

    Returns:
        Formatted string like "25.00%" or "-10.00%" or "N/A"
    """
    if rate is None:
        return "N/A"

    # Format as percentage with 2 decimal places
    # Preserve negative sign for regressions
    return f"{rate * 100.0:.2f}%"


def write_markdown_report(
    path: str | Path,
    summary: BenchmarkSummary,
) -> None:
    """Write markdown report to a file.

    This function renders the summary to markdown and writes it to the specified
    path. Parent directories are created automatically. The file is opened in
    exclusive create mode to prevent accidental overwrites.

    Args:
        path: File path (str or Path) where the report will be written
        summary: BenchmarkSummary instance to render and write

    Raises:
        TypeError: If path or summary have wrong types
        FileExistsError: If the file already exists
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a str or Path")
    if not isinstance(summary, BenchmarkSummary):
        raise TypeError("summary must be a BenchmarkSummary")

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)

    # Render markdown
    markdown = render_markdown_report(summary)

    # Write with exclusive create (prevents overwrite)
    with resolved_path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown)
