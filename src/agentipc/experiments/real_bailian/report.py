from __future__ import annotations

from collections.abc import Mapping


def render_calibration_report(summary: Mapping[str, object]) -> str:
    if not isinstance(summary, Mapping):
        raise TypeError("summary must be a Mapping[str, object]")

    status = "PASS" if summary.get("passed") is True else "FAIL"
    provider = _mapping(summary, "provider_probe")
    embedding = _mapping(summary, "embedding_probe")
    shm = _mapping(summary, "shm_probe")
    knowledge = _mapping(summary, "knowledge")
    codeact = _mapping(summary, "codeact")
    usage = _mapping(summary, "provider_usage")
    audit = _mapping(summary, "secret_audit")

    lines = [
        "# AgentIPC Real Bailian Calibration Report",
        "",
        f"Overall calibration: **{status}**",
        "",
        "## Metric semantics",
        "",
        "`runtime_memory_effective_strict` / `runtime_memory_harmful_strict` are the existing Runtime strict metrics based on exact historical-vs-final answer string equality.",
        "`validated_memory_effective` / `validated_memory_harmful` are experiment-layer ground-truth validated metrics based on deterministic task evaluators. The two definitions are intentionally kept separate; the validated metric never overwrites the Runtime metric.",
        "",
        "`text_tokens` is a communication-side token estimate from AgentIPC TextCounter. It is not Bailian billing usage. Real API usage is reported only by `llm_prompt_tokens`, `llm_completion_tokens`, and `llm_total_tokens`.",
        "",
        "## Calibration checks",
        "",
        "| Check | Result | Key evidence |",
        "|---|---|---|",
        _row("C1 LLM provider", provider, _probe_evidence(provider)),
        _row("C1 embedding", embedding, _embedding_evidence(embedding)),
        _row("C2 POSIX SHM", shm, _shm_evidence(shm)),
        _row("C3 Knowledge R2/R8", knowledge, _chain_evidence(knowledge)),
        _row("C4 CodeAct R2/R8", codeact, _chain_evidence(codeact)),
        _row("Secret audit", audit, _audit_evidence(audit)),
        "",
        "## Provider usage",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| LLM calls | {_safe(usage.get('llm_call_count'))} |",
        f"| LLM prompt tokens | {_safe(usage.get('llm_prompt_tokens'))} |",
        f"| LLM completion tokens | {_safe(usage.get('llm_completion_tokens'))} |",
        f"| LLM total tokens | {_safe(usage.get('llm_total_tokens'))} |",
        f"| Missing LLM usage records | {_safe(usage.get('llm_usage_missing_count'))} |",
        f"| Embedding calls | {_safe(usage.get('embedding_call_count'))} |",
        f"| Embedding input items | {_safe(usage.get('embedding_input_count'))} |",
        "",
        "## Knowledge memory semantics",
        "",
        _round_table(knowledge),
        "",
        "## CodeAct memory semantics",
        "",
        _round_table(codeact),
        "",
        "The calibration is a small-cost gate only. It does not run the full formal R002 experiment matrix.",
        "",
    ]
    return "\n".join(lines)


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _row(name: str, data: Mapping[str, object], evidence: str) -> str:
    passed = data.get("passed") is True
    return f"| {name} | {'PASS' if passed else 'FAIL'} | {evidence} |"


def _probe_evidence(data: Mapping[str, object]) -> str:
    return (
        f"prompt={_safe(data.get('prompt_tokens'))}, "
        f"completion={_safe(data.get('completion_tokens'))}, "
        f"latency_ms={_safe(data.get('latency_ms'))}"
    )


def _embedding_evidence(data: Mapping[str, object]) -> str:
    return (
        f"shape={_safe(data.get('shape'))}, dtype={_safe(data.get('dtype'))}, "
        f"finite={_safe(data.get('all_finite'))}"
    )


def _shm_evidence(data: Mapping[str, object]) -> str:
    return (
        f"transport={_safe(data.get('transport'))}, shape={_safe(data.get('shape'))}, "
        f"nbytes={_safe(data.get('nbytes'))}, released={_safe(data.get('released'))}"
    )


def _chain_evidence(data: Mapping[str, object]) -> str:
    rounds = data.get("rounds")
    if not isinstance(rounds, list):
        return _safe(data.get("error", "not run"))
    return f"rounds={len(rounds)}"


def _audit_evidence(data: Mapping[str, object]) -> str:
    return f"files_scanned={_safe(data.get('files_scanned'))}"


def _round_table(chain: Mapping[str, object]) -> str:
    rounds = chain.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        return f"No completed rounds. Error: {_safe(chain.get('error', 'n/a'))}"

    lines = [
        "| Round | Eval | memory_used | strict effective/harmful | validated effective/harmful | tool calls | discrepancy |",
        "|---:|---|---:|---|---|---:|---|",
    ]
    for raw in rounds:
        row = raw if isinstance(raw, Mapping) else {}
        mv = row.get("memory_validation")
        memory = mv if isinstance(mv, Mapping) else {}
        metrics_raw = row.get("metrics")
        metrics = metrics_raw if isinstance(metrics_raw, Mapping) else {}
        eval_pass = row.get("evaluation_pass") is True
        lines.append(
            "| {round} | {eval_status} | {used} | {se}/{sh} | {ve}/{vh} | {tools} | {disc} |".format(
                round=_safe(row.get("round")),
                eval_status="PASS" if eval_pass else "FAIL",
                used=_safe(memory.get("memory_used")),
                se=_safe(memory.get("runtime_memory_effective_strict")),
                sh=_safe(memory.get("runtime_memory_harmful_strict")),
                ve=_safe(memory.get("validated_memory_effective")),
                vh=_safe(memory.get("validated_memory_harmful")),
                tools=_safe(metrics.get("tool_call_count")),
                disc=_safe(memory.get("strict_vs_validated_discrepancy")),
            )
        )
    return "\n".join(lines)


def _safe(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")
