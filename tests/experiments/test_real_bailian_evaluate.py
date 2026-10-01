from agentipc.experiments.real_bailian.evaluate import (
    build_memory_validation,
    evaluate_normalized_knowledge_answer,
    normalize_answer_text,
)
from agentipc.scenarios.models import KnowledgeTask


def _task() -> KnowledgeTask:
    return KnowledgeTask.model_validate(
        {
            "group_id": "g",
            "round": 8,
            "query": "q",
            "expected": {
                "answer_contains": ["nmcli device show"],
                "evidence_ids": ["nm-basics"],
            },
            "reuse_hint": {
                "source_rounds": [2],
                "topics": ["network"],
                "expected_reuse": True,
            },
        }
    )


def test_normalization_casefold_nfkc_backticks_and_whitespace():
    assert normalize_answer_text("ＮＭＣＬＩ   `DEVICE\nSHOW`") == "nmcli device show"


def test_normalized_knowledge_evaluator_accepts_formatting_variation():
    evaluation = evaluate_normalized_knowledge_answer(
        _task(),
        "建议运行 `NMCLI   DEVICE   SHOW` 后检查 IPv4、网关和 DNS。",
    )
    assert evaluation.success is True


def test_validated_memory_effective_uses_ground_truth_result():
    result = build_memory_validation(
        {"memory_used": 1, "memory_effective": 0, "memory_harmful": 1},
        evaluator_pass=True,
    )
    assert result.validated_memory_effective == 1
    assert result.validated_memory_harmful == 0
    assert result.strict_vs_validated_discrepancy is True


def test_validated_memory_harmful_when_evaluator_fails():
    result = build_memory_validation(
        {"memory_used": 2, "memory_effective": 2, "memory_harmful": 0},
        evaluator_pass=False,
    )
    assert result.validated_memory_effective == 0
    assert result.validated_memory_harmful == 2
