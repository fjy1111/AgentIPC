from __future__ import annotations

import numpy as np

from agentipc.experiments.formal.e6_memory_fast_path import _aggregate_e6
from agentipc.experiments.formal.report import render_report
from agentipc.experiments.real_bailian.config import RealBailianConfig
from agentipc.experiments.real_bailian import runner as runner_module
from agentipc.providers.base import LLMResponse


def _config() -> RealBailianConfig:
    return RealBailianConfig(
        api_key="sk-test",
        base_url="https://example.invalid/v1",
        region="singapore",
    )


def _metrics(
    *,
    fast_hits: int = 0,
    memory_used: int = 0,
    memory_harmful: int = 0,
) -> dict[str, int | float | bool]:
    return {
        "llm_prompt_tokens": 10,
        "llm_completion_tokens": 5,
        "llm_total_tokens": 15,
        "llm_call_count": 2,
        "wire_tokens": 20,
        "wire_bytes": 80,
        "tool_call_count": 0,
        "memory_retrieved": 0,
        "memory_used": memory_used,
        "memory_effective": 0,
        "memory_harmful": memory_harmful,
        "fast_path_hit_count": fast_hits,
        "latency_ms": 10.0,
    }


def _row(
    *,
    group: str,
    config: str,
    phase: str,
    evaluation_pass: bool = True,
    runtime_success: bool = True,
    error: dict[str, str] | None = None,
    fast_hits: int = 0,
    memory_used: int = 0,
    memory_harmful: int = 0,
) -> dict[str, object]:
    return {
        "experiment": "E6",
        "group": group,
        "config": config,
        "repeat": 1,
        "sequence": 1,
        "phase": phase,
        "source_round": 1,
        "task_hash": "0" * 64,
        "runtime_success": runtime_success,
        "evaluation_pass": evaluation_pass,
        "error": error,
        "validated_fast_path_effective": int(bool(fast_hits and evaluation_pass)),
        "validated_fast_path_harmful": int(bool(fast_hits and not evaluation_pass)),
        "metrics": _metrics(
            fast_hits=fast_hits,
            memory_used=memory_used,
            memory_harmful=memory_harmful,
        ),
    }


def _healthy_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in ("knowledge", "codeact"):
        for config in ("C", "D", "D-Fast"):
            rows.append(
                _row(
                    group=group,
                    config=config,
                    phase="new",
                    memory_used=1 if config == "D" else 0,
                    memory_harmful=1 if config == "D" else 0,
                )
            )
            rows.append(
                _row(
                    group=group,
                    config=config,
                    phase="repeat",
                    fast_hits=1 if config == "D-Fast" else 0,
                )
            )
    return rows


def test_real_provider_bundle_uses_120_second_timeout(monkeypatch) -> None:
    seen: dict[str, dict[str, object]] = {}

    class FakeLLM:
        def __init__(self, **kwargs) -> None:
            seen["llm"] = kwargs

        def complete(self, messages, *, temperature=0.0):
            return LLMResponse(
                text="ok",
                prompt_tokens=1,
                completion_tokens=1,
                latency_ms=1.0,
                raw={},
            )

    class FakeEmbedding:
        def __init__(self, *, dim: int, **kwargs) -> None:
            self._dim = dim
            seen["embedding"] = {"dim": dim, **kwargs}

        @property
        def dim(self) -> int:
            return self._dim

        def embed(self, texts):
            return np.ones((len(texts), self._dim), dtype=np.float32)

    monkeypatch.setattr(runner_module, "OpenAICompatibleProvider", FakeLLM)
    monkeypatch.setattr(
        runner_module,
        "OpenAICompatibleEmbeddingProvider",
        FakeEmbedding,
    )

    runner_module.build_recording_provider_bundle(_config())

    assert runner_module.REAL_API_TIMEOUT_SEC == 120.0
    assert runner_module.REAL_API_MAX_RETRIES == 0
    assert seen["llm"]["timeout_sec"] == 120.0
    assert seen["embedding"]["timeout_sec"] == 120.0


def test_e6_timeout_invalidates_comparison_and_is_reported() -> None:
    rows = _healthy_rows()
    failed = next(
        row
        for row in rows
        if row["group"] == "knowledge"
        and row["config"] == "C"
        and row["phase"] == "repeat"
    )
    failed["runtime_success"] = False
    failed["evaluation_pass"] = False
    failed["error"] = {
        "type": "APITimeoutError",
        "message": "Request timed out.",
    }

    summary = _aggregate_e6(
        rows,
        repeat=1,
        provider={
            "llm_model": "fake-model",
            "api_region": "singapore",
            "timeout_sec": 120.0,
            "max_retries": 0,
        },
        git_sha="a" * 40,
    )

    assert summary["passed"] is False
    assert summary["infrastructure_valid"] is False
    assert summary["comparison_valid"] is False
    assert summary["infrastructure_failure_count"] == 1
    assert summary["infrastructure_failure_types"] == ["APITimeoutError"]

    knowledge_c = summary["groups"]["knowledge"]["configs"]["C"]
    assert knowledge_c["evaluation_pass_rate"] == 0.5
    assert knowledge_c["evaluation_pass_rate_excluding_infrastructure"] == 1.0
    assert knowledge_c["infrastructure_failure_count"] == 1

    report = render_report(summary)
    assert "Provider timeout: **120s**" in report
    assert "Infrastructure failures: **1**" in report
    assert "diagnostic only; comparison invalid" in report


def test_e6_baseline_answer_variation_is_not_fast_path_harm() -> None:
    rows = _healthy_rows()
    c_repeat = next(
        row
        for row in rows
        if row["group"] == "knowledge"
        and row["config"] == "C"
        and row["phase"] == "repeat"
    )
    c_repeat["evaluation_pass"] = False

    summary = _aggregate_e6(rows, repeat=1)

    assert summary["comparison_valid"] is True
    assert summary["fast_path_safety_pass"] is True
    assert summary["passed"] is True

    knowledge_d = summary["groups"]["knowledge"]["configs"]["D"]
    assert knowledge_d["strict_memory_mismatch_count"] == 1
    assert knowledge_d["strict_memory_mismatch_rate"] == 1.0
    assert knowledge_d["validated_fast_path_harmful_count"] == 0
    assert knowledge_d["validated_fast_path_harmful_rate"] == 0.0

    report = render_report(summary)
    assert "Strict memory mismatch" in report
    assert "Validated fast-path harmful" in report
    assert "it is not an evaluator failure" in report
