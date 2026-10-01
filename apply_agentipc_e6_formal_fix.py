#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

EXPECTED_BASE_PREFIX = "7dc108e8"


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise SystemExit(
            f"[agentipc-e6-formal-fix] {label}: expected old block exactly once, found {count}"
        )
    return text.replace(old, new, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    required = [
        root / "src/agentipc/experiments/real_bailian/runner.py",
        root / "src/agentipc/experiments/formal/e6_memory_fast_path.py",
        root / "src/agentipc/experiments/formal/report.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "[agentipc-e6-formal-fix] run this from the AgentIPC repository root; "
            f"missing: {missing}"
        )

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        head = "unknown"

    if head != EXPECTED_BASE_PREFIX:
        print(
            "[agentipc-e6-formal-fix] note: expected base "
            f"{EXPECTED_BASE_PREFIX}, current HEAD is {head}; exact anchors will still be checked."
        )

    runner_path = required[0]
    runner = runner_path.read_text(encoding="utf-8")
    runner = replace_once(
        runner,
        '''_PHASE = "calibration"

def run_calibration(
''',
        '''_PHASE = "calibration"
REAL_API_TIMEOUT_SEC = 120.0
# OpenAICompatibleProvider fixes max_retries=0 internally.
REAL_API_MAX_RETRIES = 0


def run_calibration(
''',
        label="real_bailian runner constants",
    )
    runner = replace_once(
        runner,
        '''    llm = OpenAICompatibleProvider(
        model=config.llm_model,
        api_key=config.api_key,
        base_url=config.base_url,
    )
    embedding = OpenAICompatibleEmbeddingProvider(
        model=config.embedding_model,
        dim=config.embedding_dim,
        api_key=config.api_key,
        base_url=config.base_url,
    )
''',
        '''    llm = OpenAICompatibleProvider(
        model=config.llm_model,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_sec=REAL_API_TIMEOUT_SEC,
    )
    embedding = OpenAICompatibleEmbeddingProvider(
        model=config.embedding_model,
        dim=config.embedding_dim,
        api_key=config.api_key,
        base_url=config.base_url,
        timeout_sec=REAL_API_TIMEOUT_SEC,
    )
''',
        label="real_bailian provider timeout",
    )

    e6_path = required[1]
    e6 = e6_path.read_text(encoding="utf-8")
    e6 = replace_once(
        e6,
        '''from __future__ import annotations

from pathlib import Path
''',
        '''from __future__ import annotations

import subprocess
from pathlib import Path
''',
        label="e6 subprocess import",
    )
    e6 = replace_once(
        e6,
        '''from agentipc.experiments.real_bailian.runner import build_recording_provider_bundle
''',
        '''from agentipc.experiments.real_bailian.runner import (
    REAL_API_MAX_RETRIES,
    REAL_API_TIMEOUT_SEC,
    build_recording_provider_bundle,
)
''',
        label="e6 runner imports",
    )
    e6 = replace_once(
        e6,
        '''_CONFIGS: tuple[tuple[str, ExperimentConfig, bool], ...] = (
    ("C", EXPERIMENT_C, False),
    ("D", EXPERIMENT_D, False),
    ("D-Fast", EXPERIMENT_D, True),
)


def run_e6(
''',
        '''_CONFIGS: tuple[tuple[str, ExperimentConfig, bool], ...] = (
    ("C", EXPERIMENT_C, False),
    ("D", EXPERIMENT_D, False),
    ("D-Fast", EXPERIMENT_D, True),
)

_INFRASTRUCTURE_ERROR_TYPES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
        "ServiceUnavailableError",
    }
)


def _read_git_sha(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    sha = completed.stdout.strip()
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        return None
    return sha


def _is_infrastructure_failure(row: dict[str, Any]) -> bool:
    if bool(row.get("runtime_success")):
        return False
    error = row.get("error")
    return (
        type(error) is dict
        and error.get("type") in _INFRASTRUCTURE_ERROR_TYPES
    )


def _comparison_valid(configs: dict[str, dict[str, Any]]) -> bool:
    return bool(
        all(item["infrastructure_failure_count"] == 0 for item in configs.values())
        and all(item["runtime_success_rate"] == 1.0 for item in configs.values())
    )


def run_e6(
''',
        label="e6 infrastructure helpers",
    )
    e6 = replace_once(
        e6,
        '''    secret = load_real_bailian_config()
    llm_recorder, embedding_recorder = build_recording_provider_bundle(secret)
    provider_bundle = ProviderBundle(llm=llm_recorder, embedding=embedding_recorder)

''',
        '''    secret = load_real_bailian_config()
    llm_recorder, embedding_recorder = build_recording_provider_bundle(secret)
    provider_bundle = ProviderBundle(llm=llm_recorder, embedding=embedding_recorder)
    provider_metadata = {
        **secret.public_fields(),
        "timeout_sec": REAL_API_TIMEOUT_SEC,
        "max_retries": REAL_API_MAX_RETRIES,
    }
    git_sha = _read_git_sha(root)

''',
        label="e6 provider metadata",
    )
    e6 = replace_once(
        e6,
        '''    return rows, _aggregate_e6(rows, repeat=repeat)
''',
        '''    return rows, _aggregate_e6(
        rows,
        repeat=repeat,
        provider=provider_metadata,
        git_sha=git_sha,
    )
''',
        label="e6 aggregate call",
    )

    old_aggregate = '''def _aggregate_e6(rows: list[dict[str, Any]], *, repeat: int) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in ("knowledge", "codeact"):
        group_rows = [row for row in rows if row["group"] == group]
        configs = {
            name: _aggregate_rows([row for row in group_rows if row["config"] == name])
            for name, _, _ in _CONFIGS
        }
        d_fast_rows = [row for row in group_rows if row["config"] == "D-Fast"]
        groups[group] = {
            "configs": configs,
            "c_vs_d_fast": _delta(configs["C"], configs["D-Fast"]),
            "d_vs_d_fast": _delta(configs["D"], configs["D-Fast"]),
            "zero_repeat_control": _aggregate_rows(
                [row for row in d_fast_rows if row["phase"] == "new"]
            ),
            "repeat_half": _aggregate_rows(
                [row for row in d_fast_rows if row["phase"] == "repeat"]
            ),
        }

    overall_configs = {
        name: _aggregate_rows([row for row in rows if row["config"] == name])
        for name, _, _ in _CONFIGS
    }
    overall_fast = [row for row in rows if row["config"] == "D-Fast"]
    zero_control = _aggregate_rows([row for row in overall_fast if row["phase"] == "new"])
    repeat_half = _aggregate_rows([row for row in overall_fast if row["phase"] == "repeat"])
    passed = bool(
        all(item["evaluation_pass_rate"] == 1.0 for item in overall_configs.values())
        and zero_control["total_fast_path_hits"] == 0
        and repeat_half["validated_fast_path_harmful_count"] == 0
    )
    return {
        "experiment": "E6",
        "repeat": repeat,
        "workload": "per group: 5 new tasks + the exact same 5 tasks repeated",
        "repeat_ratio": 0.5,
        "match_method": "exact task_topic + evaluator-validated RESULT",
        "passed": passed,
        "groups": groups,
        "overall": {
            "configs": overall_configs,
            "c_vs_d_fast": _delta(overall_configs["C"], overall_configs["D-Fast"]),
            "d_vs_d_fast": _delta(overall_configs["D"], overall_configs["D-Fast"]),
            "zero_repeat_control": zero_control,
            "repeat_half": repeat_half,
        },
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty E6 row set")
    n = len(rows)

    def total(metric: str) -> float:
        return sum(float(row["metrics"].get(metric, 0)) for row in rows)

    fast_hits = int(total("fast_path_hit_count"))
    validated_effective = sum(int(row["validated_fast_path_effective"]) for row in rows)
    validated_harmful = sum(int(row["validated_fast_path_harmful"]) for row in rows)
    return {
        "task_count": n,
        "runtime_success_rate": sum(bool(row["runtime_success"]) for row in rows) / n,
        "evaluation_pass_rate": sum(bool(row["evaluation_pass"]) for row in rows) / n,
        "total_llm_prompt_tokens": int(total("llm_prompt_tokens")),
        "total_llm_completion_tokens": int(total("llm_completion_tokens")),
        "total_llm_tokens": int(total("llm_total_tokens")),
        "total_llm_calls": int(total("llm_call_count")),
        "total_wire_tokens": int(total("wire_tokens")),
        "total_wire_bytes": int(total("wire_bytes")),
        "total_tool_calls": int(total("tool_call_count")),
        "total_memory_retrieved": int(total("memory_retrieved")),
        "total_memory_used": int(total("memory_used")),
        "total_memory_effective": int(total("memory_effective")),
        "total_memory_harmful": int(total("memory_harmful")),
        "total_fast_path_hits": fast_hits,
        "fast_path_hit_rate": fast_hits / n,
        "validated_fast_path_effective_count": validated_effective,
        "validated_fast_path_harmful_count": validated_harmful,
        "validated_fast_path_effective_rate": validated_effective / fast_hits if fast_hits else 0.0,
        "wrong_harmful_memory_rate": validated_harmful / fast_hits if fast_hits else 0.0,
        "total_latency_ms": total("latency_ms"),
        "mean_latency_ms": total("latency_ms") / n,
    }
'''

    new_aggregate = '''def _aggregate_e6(
    rows: list[dict[str, Any]],
    *,
    repeat: int,
    provider: dict[str, Any] | None = None,
    git_sha: str | None = None,
) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for group in ("knowledge", "codeact"):
        group_rows = [row for row in rows if row["group"] == group]
        configs = {
            name: _aggregate_rows([row for row in group_rows if row["config"] == name])
            for name, _, _ in _CONFIGS
        }
        d_fast_rows = [row for row in group_rows if row["config"] == "D-Fast"]
        groups[group] = {
            "configs": configs,
            "comparison_valid": _comparison_valid(configs),
            "c_vs_d_fast": _delta(configs["C"], configs["D-Fast"]),
            "d_vs_d_fast": _delta(configs["D"], configs["D-Fast"]),
            "zero_repeat_control": _aggregate_rows(
                [row for row in d_fast_rows if row["phase"] == "new"]
            ),
            "repeat_half": _aggregate_rows(
                [row for row in d_fast_rows if row["phase"] == "repeat"]
            ),
        }

    overall_configs = {
        name: _aggregate_rows([row for row in rows if row["config"] == name])
        for name, _, _ in _CONFIGS
    }
    overall_fast = [row for row in rows if row["config"] == "D-Fast"]
    zero_control = _aggregate_rows([row for row in overall_fast if row["phase"] == "new"])
    repeat_half = _aggregate_rows([row for row in overall_fast if row["phase"] == "repeat"])

    infrastructure_failure_count = sum(
        item["infrastructure_failure_count"] for item in overall_configs.values()
    )
    infrastructure_failure_types = sorted(
        {
            error_type
            for item in overall_configs.values()
            for error_type in item["infrastructure_failure_types"]
        }
    )
    infrastructure_valid = infrastructure_failure_count == 0
    comparison_valid = _comparison_valid(overall_configs)
    fast_path_safety_pass = bool(
        overall_configs["D-Fast"]["evaluation_pass_rate_excluding_infrastructure"] == 1.0
        and zero_control["total_fast_path_hits"] == 0
        and repeat_half["total_fast_path_hits"] == repeat_half["task_count"]
        and repeat_half["validated_fast_path_harmful_count"] == 0
    )
    passed = bool(comparison_valid and fast_path_safety_pass)

    return {
        "experiment": "E6",
        "repeat": repeat,
        "workload": "per group: 5 new tasks + the exact same 5 tasks repeated",
        "repeat_ratio": 0.5,
        "match_method": "exact task_topic + evaluator-validated RESULT",
        "passed": passed,
        "infrastructure_valid": infrastructure_valid,
        "infrastructure_failure_count": infrastructure_failure_count,
        "infrastructure_failure_types": infrastructure_failure_types,
        "comparison_valid": comparison_valid,
        "fast_path_safety_pass": fast_path_safety_pass,
        "provider": {} if provider is None else provider,
        "git_sha": git_sha,
        "groups": groups,
        "overall": {
            "configs": overall_configs,
            "comparison_valid": comparison_valid,
            "c_vs_d_fast": _delta(overall_configs["C"], overall_configs["D-Fast"]),
            "d_vs_d_fast": _delta(overall_configs["D"], overall_configs["D-Fast"]),
            "zero_repeat_control": zero_control,
            "repeat_half": repeat_half,
        },
    }


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate an empty E6 row set")
    n = len(rows)

    def total(metric: str) -> float:
        return sum(float(row["metrics"].get(metric, 0)) for row in rows)

    infrastructure_rows = [row for row in rows if _is_infrastructure_failure(row)]
    non_infrastructure_rows = [
        row for row in rows if not _is_infrastructure_failure(row)
    ]
    infrastructure_failure_types = sorted(
        {
            str(row["error"]["type"])
            for row in infrastructure_rows
            if type(row.get("error")) is dict and row["error"].get("type")
        }
    )

    fast_hits = int(total("fast_path_hit_count"))
    validated_effective = sum(int(row["validated_fast_path_effective"]) for row in rows)
    validated_harmful = sum(int(row["validated_fast_path_harmful"]) for row in rows)
    memory_used = int(total("memory_used"))
    strict_memory_mismatches = int(total("memory_harmful"))
    validated_harmful_rate = validated_harmful / fast_hits if fast_hits else 0.0
    non_infra_eval_rate = (
        sum(bool(row["evaluation_pass"]) for row in non_infrastructure_rows)
        / len(non_infrastructure_rows)
        if non_infrastructure_rows
        else 0.0
    )
    return {
        "task_count": n,
        "runtime_success_rate": sum(bool(row["runtime_success"]) for row in rows) / n,
        "evaluation_pass_rate": sum(bool(row["evaluation_pass"]) for row in rows) / n,
        "evaluation_pass_rate_excluding_infrastructure": non_infra_eval_rate,
        "infrastructure_failure_count": len(infrastructure_rows),
        "infrastructure_failure_rate": len(infrastructure_rows) / n,
        "infrastructure_failure_types": infrastructure_failure_types,
        "total_llm_prompt_tokens": int(total("llm_prompt_tokens")),
        "total_llm_completion_tokens": int(total("llm_completion_tokens")),
        "total_llm_tokens": int(total("llm_total_tokens")),
        "total_llm_calls": int(total("llm_call_count")),
        "total_wire_tokens": int(total("wire_tokens")),
        "total_wire_bytes": int(total("wire_bytes")),
        "total_tool_calls": int(total("tool_call_count")),
        "total_memory_retrieved": int(total("memory_retrieved")),
        "total_memory_used": memory_used,
        "total_memory_effective": int(total("memory_effective")),
        "total_memory_harmful": strict_memory_mismatches,
        "strict_memory_mismatch_count": strict_memory_mismatches,
        "strict_memory_mismatch_rate": (
            strict_memory_mismatches / memory_used if memory_used else 0.0
        ),
        "total_fast_path_hits": fast_hits,
        "fast_path_hit_rate": fast_hits / n,
        "validated_fast_path_effective_count": validated_effective,
        "validated_fast_path_harmful_count": validated_harmful,
        "validated_fast_path_effective_rate": (
            validated_effective / fast_hits if fast_hits else 0.0
        ),
        "validated_fast_path_harmful_rate": validated_harmful_rate,
        "wrong_harmful_memory_rate": validated_harmful_rate,
        "total_latency_ms": total("latency_ms"),
        "mean_latency_ms": total("latency_ms") / n,
    }
'''
    e6 = replace_once(
        e6,
        old_aggregate,
        new_aggregate,
        label="e6 aggregation",
    )

    report_path = required[2]
    report = report_path.read_text(encoding="utf-8")
    old_report = '''def _render_e6(summary: dict[str, Any]) -> str:
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
    return "\\n".join(lines) + "\\n"
'''
    new_report = '''def _render_e6(summary: dict[str, Any]) -> str:
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
    return "\\n".join(lines) + "\\n"
'''
    report = replace_once(
        report,
        old_report,
        new_report,
        label="e6 report",
    )

    test_path = root / "tests/experiments/test_e6_formal.py"
    test_content = TEST_CONTENT
    if test_path.exists():
        existing = test_path.read_text(encoding="utf-8")
        if existing != test_content:
            raise SystemExit(
                "[agentipc-e6-formal-fix] tests/experiments/test_e6_formal.py already "
                "exists with different content"
            )

    if args.check:
        print("[agentipc-e6-formal-fix] check passed; patch can be applied")
        return

    runner_path.write_text(runner, encoding="utf-8")
    e6_path.write_text(e6, encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    test_path.write_text(test_content, encoding="utf-8")
    print("[agentipc-e6-formal-fix] applied successfully")
    print("[agentipc-e6-formal-fix] modified 3 source files and added 1 offline test file")


TEST_CONTENT = 'from __future__ import annotations\n\nimport numpy as np\n\nfrom agentipc.experiments.formal.e6_memory_fast_path import _aggregate_e6\nfrom agentipc.experiments.formal.report import render_report\nfrom agentipc.experiments.real_bailian.config import RealBailianConfig\nfrom agentipc.experiments.real_bailian import runner as runner_module\nfrom agentipc.providers.base import LLMResponse\n\n\ndef _config() -> RealBailianConfig:\n    return RealBailianConfig(\n        api_key="sk-test",\n        base_url="https://example.invalid/v1",\n        region="singapore",\n    )\n\n\ndef _metrics(\n    *,\n    fast_hits: int = 0,\n    memory_used: int = 0,\n    memory_harmful: int = 0,\n) -> dict[str, int | float | bool]:\n    return {\n        "llm_prompt_tokens": 10,\n        "llm_completion_tokens": 5,\n        "llm_total_tokens": 15,\n        "llm_call_count": 2,\n        "wire_tokens": 20,\n        "wire_bytes": 80,\n        "tool_call_count": 0,\n        "memory_retrieved": 0,\n        "memory_used": memory_used,\n        "memory_effective": 0,\n        "memory_harmful": memory_harmful,\n        "fast_path_hit_count": fast_hits,\n        "latency_ms": 10.0,\n    }\n\n\ndef _row(\n    *,\n    group: str,\n    config: str,\n    phase: str,\n    evaluation_pass: bool = True,\n    runtime_success: bool = True,\n    error: dict[str, str] | None = None,\n    fast_hits: int = 0,\n    memory_used: int = 0,\n    memory_harmful: int = 0,\n) -> dict[str, object]:\n    return {\n        "experiment": "E6",\n        "group": group,\n        "config": config,\n        "repeat": 1,\n        "sequence": 1,\n        "phase": phase,\n        "source_round": 1,\n        "task_hash": "0" * 64,\n        "runtime_success": runtime_success,\n        "evaluation_pass": evaluation_pass,\n        "error": error,\n        "validated_fast_path_effective": int(bool(fast_hits and evaluation_pass)),\n        "validated_fast_path_harmful": int(bool(fast_hits and not evaluation_pass)),\n        "metrics": _metrics(\n            fast_hits=fast_hits,\n            memory_used=memory_used,\n            memory_harmful=memory_harmful,\n        ),\n    }\n\n\ndef _healthy_rows() -> list[dict[str, object]]:\n    rows: list[dict[str, object]] = []\n    for group in ("knowledge", "codeact"):\n        for config in ("C", "D", "D-Fast"):\n            rows.append(\n                _row(\n                    group=group,\n                    config=config,\n                    phase="new",\n                    memory_used=1 if config == "D" else 0,\n                    memory_harmful=1 if config == "D" else 0,\n                )\n            )\n            rows.append(\n                _row(\n                    group=group,\n                    config=config,\n                    phase="repeat",\n                    fast_hits=1 if config == "D-Fast" else 0,\n                )\n            )\n    return rows\n\n\ndef test_real_provider_bundle_uses_120_second_timeout(monkeypatch) -> None:\n    seen: dict[str, dict[str, object]] = {}\n\n    class FakeLLM:\n        def __init__(self, **kwargs) -> None:\n            seen["llm"] = kwargs\n\n        def complete(self, messages, *, temperature=0.0):\n            return LLMResponse(\n                text="ok",\n                prompt_tokens=1,\n                completion_tokens=1,\n                latency_ms=1.0,\n                raw={},\n            )\n\n    class FakeEmbedding:\n        def __init__(self, *, dim: int, **kwargs) -> None:\n            self._dim = dim\n            seen["embedding"] = {"dim": dim, **kwargs}\n\n        @property\n        def dim(self) -> int:\n            return self._dim\n\n        def embed(self, texts):\n            return np.ones((len(texts), self._dim), dtype=np.float32)\n\n    monkeypatch.setattr(runner_module, "OpenAICompatibleProvider", FakeLLM)\n    monkeypatch.setattr(\n        runner_module,\n        "OpenAICompatibleEmbeddingProvider",\n        FakeEmbedding,\n    )\n\n    runner_module.build_recording_provider_bundle(_config())\n\n    assert runner_module.REAL_API_TIMEOUT_SEC == 120.0\n    assert runner_module.REAL_API_MAX_RETRIES == 0\n    assert seen["llm"]["timeout_sec"] == 120.0\n    assert seen["embedding"]["timeout_sec"] == 120.0\n\n\ndef test_e6_timeout_invalidates_comparison_and_is_reported() -> None:\n    rows = _healthy_rows()\n    failed = next(\n        row\n        for row in rows\n        if row["group"] == "knowledge"\n        and row["config"] == "C"\n        and row["phase"] == "repeat"\n    )\n    failed["runtime_success"] = False\n    failed["evaluation_pass"] = False\n    failed["error"] = {\n        "type": "APITimeoutError",\n        "message": "Request timed out.",\n    }\n\n    summary = _aggregate_e6(\n        rows,\n        repeat=1,\n        provider={\n            "llm_model": "fake-model",\n            "api_region": "singapore",\n            "timeout_sec": 120.0,\n            "max_retries": 0,\n        },\n        git_sha="a" * 40,\n    )\n\n    assert summary["passed"] is False\n    assert summary["infrastructure_valid"] is False\n    assert summary["comparison_valid"] is False\n    assert summary["infrastructure_failure_count"] == 1\n    assert summary["infrastructure_failure_types"] == ["APITimeoutError"]\n\n    knowledge_c = summary["groups"]["knowledge"]["configs"]["C"]\n    assert knowledge_c["evaluation_pass_rate"] == 0.5\n    assert knowledge_c["evaluation_pass_rate_excluding_infrastructure"] == 1.0\n    assert knowledge_c["infrastructure_failure_count"] == 1\n\n    report = render_report(summary)\n    assert "Provider timeout: **120s**" in report\n    assert "Infrastructure failures: **1**" in report\n    assert "diagnostic only; comparison invalid" in report\n\n\ndef test_e6_baseline_answer_variation_is_not_fast_path_harm() -> None:\n    rows = _healthy_rows()\n    c_repeat = next(\n        row\n        for row in rows\n        if row["group"] == "knowledge"\n        and row["config"] == "C"\n        and row["phase"] == "repeat"\n    )\n    c_repeat["evaluation_pass"] = False\n\n    summary = _aggregate_e6(rows, repeat=1)\n\n    assert summary["comparison_valid"] is True\n    assert summary["fast_path_safety_pass"] is True\n    assert summary["passed"] is True\n\n    knowledge_d = summary["groups"]["knowledge"]["configs"]["D"]\n    assert knowledge_d["strict_memory_mismatch_count"] == 1\n    assert knowledge_d["strict_memory_mismatch_rate"] == 1.0\n    assert knowledge_d["validated_fast_path_harmful_count"] == 0\n    assert knowledge_d["validated_fast_path_harmful_rate"] == 0.0\n\n    report = render_report(summary)\n    assert "Strict memory mismatch" in report\n    assert "Validated fast-path harmful" in report\n    assert "it is not an evaluator failure" in report\n'


if __name__ == "__main__":
    main()
