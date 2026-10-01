from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping

from agentipc.experiments.real_bailian.models import (
    MemoryValidation,
    NormalizedKnowledgeEvaluation,
)
from agentipc.scenarios.models import KnowledgeTask


def normalize_answer_text(value: str) -> str:
    """Normalize a real-model answer without using another model as judge."""
    if not isinstance(value, str):
        raise TypeError("value must be a str")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = normalized.replace("`", "")
    return re.sub(r"\s+", " ", normalized).strip()


def evaluate_normalized_knowledge_answer(
    task: KnowledgeTask,
    answer: str,
) -> NormalizedKnowledgeEvaluation:
    if not isinstance(task, KnowledgeTask):
        raise TypeError("task must be a KnowledgeTask")
    if not isinstance(answer, str):
        raise TypeError("answer must be a str")

    normalized_answer = normalize_answer_text(answer)
    matched: list[str] = []
    missing: list[str] = []
    for expected_fragment in task.expected.answer_contains:
        normalized_fragment = normalize_answer_text(expected_fragment)
        if normalized_fragment in normalized_answer:
            matched.append(expected_fragment)
        else:
            missing.append(expected_fragment)

    return NormalizedKnowledgeEvaluation(
        success=not missing,
        matched_answer_contains=matched,
        missing_answer_contains=missing,
    )


def build_memory_validation(
    metrics: Mapping[str, object],
    *,
    evaluator_pass: bool,
) -> MemoryValidation:
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics must be a Mapping[str, object]")
    if type(evaluator_pass) is not bool:
        raise TypeError("evaluator_pass must be a bool")

    memory_used = _metric_int(metrics, "memory_used")
    strict_effective = _metric_int(metrics, "memory_effective")
    strict_harmful = _metric_int(metrics, "memory_harmful")

    if memory_used > 0 and evaluator_pass:
        validated_effective = memory_used
        validated_harmful = 0
    elif memory_used > 0:
        validated_effective = 0
        validated_harmful = memory_used
    else:
        validated_effective = 0
        validated_harmful = 0

    return MemoryValidation(
        memory_used=memory_used,
        runtime_memory_effective_strict=strict_effective,
        runtime_memory_harmful_strict=strict_harmful,
        validated_memory_effective=validated_effective,
        validated_memory_harmful=validated_harmful,
        strict_vs_validated_discrepancy=(
            strict_effective != validated_effective
            or strict_harmful != validated_harmful
        ),
    )


def _metric_int(metrics: Mapping[str, object], name: str) -> int:
    value = metrics.get(name, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"metrics[{name!r}] must be a non-negative int")
    return value
