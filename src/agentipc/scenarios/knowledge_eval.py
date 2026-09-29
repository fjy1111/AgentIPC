from __future__ import annotations

from pydantic import BaseModel, ConfigDict, StrictBool

from agentipc.scenarios.models import KnowledgeTask


class KnowledgeEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: StrictBool
    matched_answer_contains: list[str]
    missing_answer_contains: list[str]


def evaluate_knowledge_answer(
    task: KnowledgeTask,
    answer: str,
) -> KnowledgeEvaluation:
    if not isinstance(task, KnowledgeTask):
        raise TypeError("task must be a KnowledgeTask")
    if type(answer) is not str:
        raise TypeError("answer must be a str")

    matched: list[str] = []
    missing: list[str] = []
    for expected_fragment in task.expected.answer_contains:
        if expected_fragment in answer:
            matched.append(expected_fragment)
        else:
            missing.append(expected_fragment)

    return KnowledgeEvaluation(
        success=not missing,
        matched_answer_contains=matched,
        missing_answer_contains=missing,
    )
