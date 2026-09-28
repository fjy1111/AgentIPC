from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
import unicodedata

import numpy as np

from agentipc.agents.base import BaseAgent
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.state.plan_vector import (
    PLAN_VECTOR_DIM,
    PLAN_VECTOR_KIND,
    encode_plan_vector,
)

if TYPE_CHECKING:
    from agentipc.runtime.context import RunContext


_SEMANTIC_WEIGHT = 0.60
_KEYWORD_WEIGHT = 0.40
_STATE_WEIGHT = 0.50
_DEFAULT_TOP_K = 3

_ALLOWED_KNOWLEDGE_KEYS = frozenset(
    {
        "document_id",
        "text",
        "keywords",
        "state_profile",
    }
)


@dataclass(frozen=True, slots=True)
class _KnowledgeItem:
    document_id: str
    text: str
    keywords: tuple[str, ...]
    search_text: str
    tokens: frozenset[str]
    state_vector: np.ndarray | None


class RetrieverAgent(BaseAgent):
    agent_id = "retriever"
    capabilities: list[str] = []

    def __init__(
        self,
        knowledge: list[dict[str, object]],
    ) -> None:
        if type(knowledge) is not list:
            raise TypeError("knowledge must be a list[dict[str, object]]")

        items: list[_KnowledgeItem] = []
        seen_document_ids: set[str] = set()

        for raw_item in knowledge:
            if type(raw_item) is not dict:
                raise TypeError("knowledge items must be dict[str, object]")

            unknown_keys = set(raw_item) - _ALLOWED_KNOWLEDGE_KEYS
            if unknown_keys:
                unknown = ", ".join(sorted(repr(key) for key in unknown_keys))
                raise ValueError(f"knowledge item contains unknown fields: {unknown}")

            if "document_id" not in raw_item:
                raise ValueError("knowledge item requires document_id")
            document_id = raw_item["document_id"]
            if type(document_id) is not str:
                raise TypeError("document_id must be a str")
            if document_id == "":
                raise ValueError("document_id must be non-empty")
            if document_id in seen_document_ids:
                raise ValueError(f"duplicate document_id: {document_id!r}")
            seen_document_ids.add(document_id)

            if "text" not in raw_item:
                raise ValueError("knowledge item requires text")
            text = raw_item["text"]
            if type(text) is not str:
                raise TypeError("text must be a str")
            if text == "":
                raise ValueError("text must be non-empty")

            raw_keywords = raw_item.get("keywords", [])
            if type(raw_keywords) is not list:
                raise TypeError("keywords must be a list[str]")
            keywords: list[str] = []
            for keyword in raw_keywords:
                if type(keyword) is not str:
                    raise TypeError("keywords items must be str")
                if keyword == "":
                    raise ValueError("keywords items must be non-empty")
                keywords.append(keyword)

            state_profile = raw_item.get("state_profile")
            if state_profile is not None:
                if type(state_profile) is not dict:
                    raise TypeError("state_profile must be a dict or None")
                state_vector = encode_plan_vector(state_profile)
            else:
                state_vector = None

            search_text = text + " " + " ".join(keywords)
            items.append(
                _KnowledgeItem(
                    document_id=document_id,
                    text=text,
                    keywords=tuple(keywords),
                    search_text=search_text,
                    tokens=frozenset(_tokens(search_text)),
                    state_vector=state_vector,
                )
            )

        self._knowledge = tuple(items)

    def handle(
        self,
        envelope: AgentEnvelope,
        ctx: "RunContext",
    ) -> AgentEnvelope:
        if envelope.message_type is not MessageType.REQUEST:
            raise ValueError("retriever only accepts REQUEST envelopes")
        if envelope.action is not ActionType.RETRIEVE:
            raise ValueError("retriever only accepts RETRIEVE actions")

        if "plan" not in envelope.args:
            raise ValueError("retriever requires args['plan']")
        plan = envelope.args["plan"]
        if type(plan) is not dict:
            raise TypeError("plan must be a dict")

        if "task" not in plan:
            raise ValueError("plan requires task")
        task = plan["task"]
        if type(task) is not str:
            raise TypeError("plan task must be a str")
        if task == "":
            raise ValueError("plan task must be non-empty")

        if "retrieval_topics" in plan:
            raw_topics = plan["retrieval_topics"]
            if type(raw_topics) is not list:
                raise TypeError("retrieval_topics must be a list[str]")
            retrieval_topics: list[str] = []
            for topic in raw_topics:
                if type(topic) is not str:
                    raise TypeError("retrieval_topics items must be str")
                if topic == "":
                    raise ValueError("retrieval_topics items must be non-empty")
                retrieval_topics.append(topic)
        else:
            retrieval_topics = [task]

        top_k = envelope.args.get("top_k", _DEFAULT_TOP_K)
        if type(top_k) is not int:
            raise TypeError("top_k must be an int")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")

        query = " ".join(retrieval_topics) if retrieval_topics else task

        plan_refs = [
            ref
            for ref in envelope.state_refs
            if ref.kind == PLAN_VECTOR_KIND
        ]
        if len(plan_refs) > 1:
            raise ValueError("retriever accepts at most one plan_vector StateRef")

        resolved_plan_vector: np.ndarray | None = None
        if plan_refs:
            resolved_plan_vector = ctx.state_hub.resolve_array(plan_refs[0])
            _validate_plan_vector(resolved_plan_vector)

        embedding = ctx.provider_bundle.embedding
        texts = [query, *(item.search_text for item in self._knowledge)]
        matrix = embedding.embed(texts)
        _validate_embedding_matrix(
            matrix,
            expected_rows=1 + len(self._knowledge),
            expected_dim=embedding.dim,
        )

        query_vector = matrix[0]
        query_tokens = _tokens(query)
        candidates: list[tuple[float, str, str]] = []

        for index, item in enumerate(self._knowledge, start=1):
            semantic_score = _cosine_score(
                query_vector,
                matrix[index],
            )
            keyword_score = (
                len(query_tokens & item.tokens) / len(query_tokens)
                if query_tokens
                else 0.0
            )
            baseline_score = (
                _SEMANTIC_WEIGHT * semantic_score
                + _KEYWORD_WEIGHT * keyword_score
            )

            if resolved_plan_vector is None or item.state_vector is None:
                state_similarity = 0.0
            else:
                state_similarity = _cosine_score(
                    resolved_plan_vector,
                    item.state_vector,
                )

            final_score = baseline_score
            if resolved_plan_vector is not None:
                final_score += _STATE_WEIGHT * state_similarity

            if not np.isfinite(final_score):
                raise ValueError("retrieval score must be finite")
            if final_score > 0.0:
                candidates.append(
                    (
                        float(final_score),
                        item.document_id,
                        item.text,
                    )
                )

        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1]))
        evidence = [
            {
                "document_id": document_id,
                "text": text,
                "score": score,
            }
            for score, document_id, text in candidates[:top_k]
        ]

        return AgentEnvelope(
            trace_id=envelope.trace_id,
            task_id=envelope.task_id,
            step_id=envelope.step_id,
            sender=self.agent_id,
            receiver=envelope.sender,
            message_type=MessageType.RESULT,
            action=ActionType.RETRIEVE,
            capability="retrieve",
            result={"evidence": evidence},
            status=MessageStatus.OK,
            state_refs=[],
            artifact_refs=[],
            memory_refs=[],
        )


def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return set(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _validate_embedding_matrix(
    matrix: object,
    *,
    expected_rows: int,
    expected_dim: object,
) -> None:
    if not isinstance(matrix, np.ndarray):
        raise ValueError("embedding provider must return a numpy.ndarray")
    if matrix.ndim != 2:
        raise ValueError("embedding provider output must be 2-D")
    if matrix.shape[0] != expected_rows:
        raise ValueError("embedding provider returned an unexpected row count")
    if matrix.shape[1] != expected_dim:
        raise ValueError("embedding provider returned an unexpected dimension")
    if (
        not np.issubdtype(matrix.dtype, np.number)
        or np.issubdtype(matrix.dtype, np.complexfloating)
    ):
        raise ValueError("embedding provider output must be real numeric")
    try:
        finite = bool(np.all(np.isfinite(matrix)))
    except TypeError as exc:
        raise ValueError("embedding provider output must be finite") from exc
    if not finite:
        raise ValueError("embedding provider output must be finite")


def _validate_plan_vector(vector: object) -> None:
    if not isinstance(vector, np.ndarray):
        raise ValueError("resolved plan state must be a numpy.ndarray")
    if vector.shape != (PLAN_VECTOR_DIM,):
        raise ValueError(
            f"resolved plan state must have shape ({PLAN_VECTOR_DIM},)"
        )
    if vector.dtype != np.dtype(np.float32):
        raise ValueError("resolved plan state must have dtype float32")
    if not bool(np.all(np.isfinite(vector))):
        raise ValueError("resolved plan state must be finite")
    if float(np.linalg.norm(vector)) <= 0.0:
        raise ValueError("resolved plan state must have non-zero norm")


def _cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    cosine = float(np.dot(left, right) / (left_norm * right_norm))
    if not np.isfinite(cosine):
        raise ValueError("cosine similarity must be finite")
    return min(1.0, max(0.0, cosine))
