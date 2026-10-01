from __future__ import annotations

from pathlib import Path

from agentipc.experiments.formal import e2_knowledge, e3_codeact


def _round(round_number: int, *, memory: bool, tool_calls: int) -> dict:
    return {
        "round": round_number,
        "evaluation_pass": True,
        "execution_operation": "identity" if memory else "codeact",
        "metrics": {
            "latency_ms": 100.0 if not memory else 50.0,
            "llm_total_tokens": 200 if not memory else 100,
            "protocol_bytes": 1000,
            "state_bytes": 256,
            "tool_call_count": tool_calls,
            "repeated_tool_call_count": 0,
            "memory_retrieved": 1 if memory else 0,
            "memory_used": 1 if memory else 0,
        },
        "memory_validation": {
            "runtime_memory_effective_strict": 1 if memory else 0,
            "validated_memory_effective": 1 if memory else 0,
        },
    }


def _patch_common(monkeypatch, module, mini_name: str) -> list[Path]:
    seen_dirs: list[Path] = []
    monkeypatch.setattr(module, "load_real_bailian_config", lambda: object())
    monkeypatch.setattr(
        module,
        "build_recording_provider_bundle",
        lambda _secret: ("llm-recorder", "embedding-recorder"),
    )

    def fake_mini(**kwargs):
        bundle = kwargs["provider_bundle"]
        assert bundle.llm == "llm-recorder"
        assert bundle.embedding == "embedding-recorder"
        result_dir = kwargs["result_dir"]
        seen_dirs.append(result_dir)
        summary = {
            "passed": True,
            "rounds": [
                _round(2, memory=False, tool_calls=1),
                _round(8, memory=True, tool_calls=0),
            ],
        }
        return summary, [
            {"scenario": mini_name, **summary["rounds"][0]},
            {"scenario": mini_name, **summary["rounds"][1]},
        ]

    monkeypatch.setattr(module, mini_name, fake_mini)
    return seen_dirs


def test_e2_builds_provider_bundle_and_isolates_repeats(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen = _patch_common(
        monkeypatch,
        e2_knowledge,
        "run_knowledge_mini_calibration",
    )
    rows, summary = e2_knowledge.run_e2(
        root=tmp_path,
        result_dir=tmp_path / "result",
        repeat=2,
    )

    assert len(rows) == 4
    assert [row["repeat"] for row in rows] == [1, 1, 2, 2]
    assert seen[0] != seen[1]
    assert summary["passed"] is True
    assert summary["round8"]["mean_memory_used"] == 1.0
    assert summary["round8"]["validated_memory_effective_rate"] == 1.0


def test_e3_builds_provider_bundle_and_isolates_repeats(
    monkeypatch,
    tmp_path: Path,
) -> None:
    seen = _patch_common(
        monkeypatch,
        e3_codeact,
        "run_codeact_mini_calibration",
    )
    rows, summary = e3_codeact.run_e3(
        root=tmp_path,
        result_dir=tmp_path / "result",
        repeat=3,
    )

    assert len(rows) == 6
    assert len(set(seen)) == 3
    assert summary["passed"] is True
    assert summary["round2"]["mean_tool_calls"] == 1.0
    assert summary["round8"]["mean_tool_calls"] == 0.0
    assert summary["round8"]["identity_operation_rate"] == 1.0
    assert summary["delta"]["tool_call_reduction_pct"] == 100.0
