from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


StrictString = Annotated[str, Field(strict=True)]


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: StrictString
    title: StrictString
    body: StrictString
    tags: list[StrictString]

    @field_validator("document_id", "title", "body")
    @classmethod
    def _require_non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must be a non-empty string")
        return value

    @field_validator("tags", mode="before")
    @classmethod
    def _require_tag_list(cls, value: object) -> object:
        if not isinstance(value, list):
            raise ValueError("tags must be a list[str]")
        if not value:
            raise ValueError("tags must contain at least one item")
        return value

    @field_validator("tags")
    @classmethod
    def _validate_tags(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        for tag in value:
            stripped = tag.strip()
            if not stripped:
                raise ValueError("tags must contain only non-empty strings")
            normalized.append(stripped)
        if len(normalized) != len(set(normalized)):
            raise ValueError("tags must not contain duplicates")
        return value


class _KnowledgeMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: StrictString
    title: StrictString
    tags: list[StrictString]


def _parse_markdown_document(path: Path) -> KnowledgeDocument:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    if not lines or lines[0] != "---":
        raise ValueError(f"{path.name}: first line must be '---'")

    try:
        closing_index = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path.name}: front matter is not closed") from exc

    metadata_raw = yaml.safe_load("\n".join(lines[1:closing_index]))
    if not isinstance(metadata_raw, dict):
        raise ValueError(f"{path.name}: front matter must be a mapping")

    metadata = _KnowledgeMetadata.model_validate(metadata_raw)
    body = "\n".join(lines[closing_index + 1 :])

    return KnowledgeDocument(
        document_id=metadata.document_id,
        title=metadata.title,
        body=body,
        tags=metadata.tags,
    )


def load_knowledge_documents(root: str | Path) -> list[KnowledgeDocument]:
    if not isinstance(root, (str, Path)):
        raise TypeError("root must be str or pathlib.Path")

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)
    if not root_path.is_dir():
        raise NotADirectoryError(root_path)

    markdown_files = sorted(root_path.glob("*.md"), key=lambda path: path.name)
    if not markdown_files:
        raise ValueError("knowledge directory must contain at least one .md file")

    documents: list[KnowledgeDocument] = []
    seen_ids: set[str] = set()
    for path in markdown_files:
        document = _parse_markdown_document(path)
        if document.document_id in seen_ids:
            raise ValueError(f"duplicate document_id: {document.document_id}")
        seen_ids.add(document.document_id)
        documents.append(document)

    return documents
