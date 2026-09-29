from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


StrictString = Annotated[str, Field(strict=True)]
PositiveStrictInt = Annotated[int, Field(ge=1, strict=True)]
StrictBoolean = Annotated[bool, Field(strict=True)]


def _require_non_empty(value: str) -> str:
    if not value.strip():
        raise ValueError("value must be a non-empty string")
    return value


def _require_list(value: object, field_name: str) -> object:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    return value


def _validate_unique_non_empty_strings(values: list[str], field_name: str) -> list[str]:
    normalized: list[str] = []
    for item in values:
        stripped = item.strip()
        if not stripped:
            raise ValueError(f"{field_name} must contain only non-empty strings")
        normalized.append(stripped)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class KnowledgeExpected(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_contains: list[StrictString]
    evidence_ids: list[StrictString]

    @field_validator("answer_contains", "evidence_ids", mode="before")
    @classmethod
    def _require_lists(cls, value: object, info) -> object:
        value = _require_list(value, info.field_name)
        if not value:
            raise ValueError(f"{info.field_name} must contain at least one item")
        return value

    @field_validator("answer_contains", "evidence_ids")
    @classmethod
    def _validate_string_lists(cls, value: list[str], info) -> list[str]:
        return _validate_unique_non_empty_strings(value, info.field_name)


class ReuseHint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_rounds: list[PositiveStrictInt]
    topics: list[StrictString]
    expected_reuse: StrictBoolean

    @field_validator("source_rounds", "topics", mode="before")
    @classmethod
    def _require_lists(cls, value: object, info) -> object:
        return _require_list(value, info.field_name)

    @field_validator("source_rounds")
    @classmethod
    def _validate_source_rounds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("source_rounds must not contain duplicates")
        return value

    @field_validator("topics")
    @classmethod
    def _validate_topics(cls, value: list[str]) -> list[str]:
        return _validate_unique_non_empty_strings(value, "topics")

    @model_validator(mode="after")
    def _require_source_for_reuse(self) -> "ReuseHint":
        if self.expected_reuse and not self.source_rounds:
            raise ValueError("expected_reuse=True requires at least one source round")
        return self


class KnowledgeTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: StrictString
    round: PositiveStrictInt
    query: StrictString
    expected: KnowledgeExpected
    reuse_hint: ReuseHint

    @field_validator("group_id", "query")
    @classmethod
    def _require_non_empty_strings(cls, value: str) -> str:
        return _require_non_empty(value)

    @model_validator(mode="after")
    def _validate_source_rounds(self) -> "KnowledgeTask":
        invalid = [source for source in self.reuse_hint.source_rounds if source >= self.round]
        if invalid:
            raise ValueError("reuse source rounds must be strictly less than task round")
        return self


def load_knowledge_tasks(path: str | Path) -> list[KnowledgeTask]:
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be str or pathlib.Path")

    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if not isinstance(raw, list):
        raise ValueError("knowledge task JSON top level must be a list")
    if not raw:
        raise ValueError("knowledge task list must not be empty")

    tasks: list[KnowledgeTask] = []
    seen_keys: set[tuple[str, int]] = set()
    for item in raw:
        task = KnowledgeTask.model_validate(item)
        key = (task.group_id, task.round)
        if key in seen_keys:
            raise ValueError(f"duplicate knowledge task: {key}")
        seen_keys.add(key)
        tasks.append(task)

    return tasks
