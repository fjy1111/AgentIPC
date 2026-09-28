from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from agentipc.agents.base import BaseAgent
from agentipc.agents.retriever import RetrieverAgent
from agentipc.protocol.codec import decode, encode
from agentipc.protocol.envelope import AgentEnvelope
from agentipc.protocol.enums import ActionType, MessageStatus, MessageType
from agentipc.providers.hash_embedding import HashEmbeddingProvider


class ForbiddenService:
    def __getattr__(self, name: str):
        raise AssertionError(f"T082 must not access forbidden service: {name}")


class RecordingEmbeddingProvider:
    def __init__(self, delegate: HashEmbeddingProvider) -> None:
        self.delegate = delegate
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self.delegate.dim

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return self.delegate.embed(texts)


class BadEmbeddingProvider:
    def __init__(self, value: object, dim: int = 128) -> None:
        self.value = value
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]):
        return self.value


def _knowledge() -> list[dict[str, object]]:
    return [
        {
            "document_id": "network-manager",
            "text": (
                "Use NetworkManager tools to inspect openEuler "
                "network connectivity and active connections."
            ),
            "keywords": ["NetworkManager", "network connectivity", "openEuler"],
        },
        {
            "document_id": "filesystem-repair",
            "text": "Use fsck carefully when diagnosing filesystem corruption.",
            "keywords": ["filesystem", "repair"],
        },
        {
            "document_id": "package-management",
            "text": "DNF manages RPM packages and repositories on openEuler.",
            "keywords": ["dnf", "rpm", "packages"],
        },
    ]


def _request(
    *,
    plan: object | None = None,
    top_k: object = 3,
) -> AgentEnvelope:
    if plan is None:
        plan = {
            "task": "diagnose openEuler network connectivity",
            "retrieval_topics": [
                "NetworkManager",
                "network connectivity",
            ],
        }
    return AgentEnvelope(
        message_id="msg-retrieve",
        trace_id="trace-retrieve",
        task_id="task-retrieve",
        step_id="step-retrieve",
        sender="runtime",
        receiver="retriever",
        message_type=MessageType.REQUEST,
        action=ActionType.RETRIEVE,
        args={"plan": plan, "top_k": top_k},
        metrics={"sentinel": "unchanged"},
    )


def _context(embedding):
    return SimpleNamespace(
        provider_bundle=SimpleNamespace(
            embedding=embedding,
            llm=ForbiddenService(),
        ),
        state_hub=ForbiddenService(),
        memory_service=ForbiddenService(),
        artifact_store=ForbiddenService(),
        metrics=ForbiddenService(),
        trace_logger=ForbiddenService(),
    )


def test_basic_retrieval_uses_real_embedding_and_returns_expected_envelope() -> None:
    knowledge = _knowledge()
    knowledge_before = deepcopy(knowledge)
    embedding = RecordingEmbeddingProvider(HashEmbeddingProvider())
    ctx = _context(embedding)
    request = _request()
    request_before = request.model_copy(deep=True)

    output = RetrieverAgent(knowledge).handle(request, ctx)

    assert isinstance(RetrieverAgent(knowledge), BaseAgent)
    assert RetrieverAgent.agent_id == "retriever"
    assert RetrieverAgent.capabilities == []

    assert len(embedding.calls) == 1
    assert embedding.calls[0][0] == "NetworkManager network connectivity"
    assert len(embedding.calls[0]) == 1 + len(knowledge)
    assert "NetworkManager" in embedding.calls[0][1]
    assert "filesystem" in embedding.calls[0][2]
    assert "DNF" in embedding.calls[0][3]

    assert output.trace_id == request.trace_id
    assert output.task_id == request.task_id
    assert output.step_id == request.step_id
    assert output.sender == "retriever"
    assert output.receiver == "runtime"
    assert output.message_type is MessageType.RESULT
    assert output.action is ActionType.RETRIEVE
    assert output.status is MessageStatus.OK
    assert output.capability == "retrieve"
    assert output.args == {}
    assert output.state_refs == []
    assert output.artifact_refs == []
    assert output.memory_refs == []

    evidence = output.result["evidence"]
    assert evidence[0]["document_id"] == "network-manager"
    assert evidence[0]["text"] == knowledge[0]["text"]
    assert isinstance(evidence[0]["score"], float)
    assert np.isfinite(evidence[0]["score"])
    assert set(evidence[0]) == {"document_id", "text", "score"}

    assert request == request_before
    assert knowledge == knowledge_before
    assert decode(encode(output)) == output


def test_basic_retrieval_is_deterministic() -> None:
    agent = RetrieverAgent(_knowledge())
    request = _request()
    first = agent.handle(request, _context(HashEmbeddingProvider()))
    second = agent.handle(request, _context(HashEmbeddingProvider()))

    first_evidence = first.result["evidence"]
    second_evidence = second.result["evidence"]
    assert [item["document_id"] for item in first_evidence] == [
        item["document_id"] for item in second_evidence
    ]
    assert [item["score"] for item in first_evidence] == pytest.approx(
        [item["score"] for item in second_evidence]
    )


def test_missing_retrieval_topics_defaults_to_task() -> None:
    embedding = RecordingEmbeddingProvider(HashEmbeddingProvider())
    request = _request(
        plan={"task": "NetworkManager network connectivity"},
    )
    RetrieverAgent(_knowledge()).handle(request, _context(embedding))
    assert embedding.calls[0][0] == "NetworkManager network connectivity"


def test_explicit_empty_retrieval_topics_falls_back_to_task() -> None:
    embedding = RecordingEmbeddingProvider(HashEmbeddingProvider())
    request = _request(
        plan={
            "task": "NetworkManager network connectivity",
            "retrieval_topics": [],
        }
    )
    RetrieverAgent(_knowledge()).handle(request, _context(embedding))
    assert embedding.calls[0][0] == "NetworkManager network connectivity"


def test_top_k_is_applied_after_sorting() -> None:
    output = RetrieverAgent(_knowledge()).handle(
        _request(top_k=1),
        _context(HashEmbeddingProvider()),
    )
    assert len(output.result["evidence"]) == 1
    assert output.result["evidence"][0]["document_id"] == "network-manager"


def test_zero_score_candidates_are_omitted() -> None:
    provider = BadEmbeddingProvider(np.zeros((2, 4), dtype=np.float32), dim=4)
    knowledge = [{"document_id": "x", "text": "alpha", "keywords": []}]
    request = _request(plan={"task": "!!!", "retrieval_topics": ["!!!"]})
    output = RetrieverAgent(knowledge).handle(request, _context(provider))
    assert output.result == {"evidence": []}


def test_constructor_rejects_duplicate_document_id() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        RetrieverAgent(
            [
                {"document_id": "same", "text": "a"},
                {"document_id": "same", "text": "b"},
            ]
        )


@pytest.mark.parametrize(
    ("knowledge", "error"),
    [
        (None, TypeError),
        ([{"document_id": 1, "text": "x"}], TypeError),
        ([{"document_id": "", "text": "x"}], ValueError),
        ([{"document_id": "x", "text": 1}], TypeError),
        ([{"document_id": "x", "text": ""}], ValueError),
        ([{"document_id": "x", "text": "x", "keywords": "k"}], TypeError),
        ([{"document_id": "x", "text": "x", "keywords": [1]}], TypeError),
        ([{"document_id": "x", "text": "x", "keywords": [""]}], ValueError),
        ([{"document_id": "x", "text": "x", "extra": 1}], ValueError),
    ],
)
def test_constructor_validates_knowledge_schema(knowledge, error) -> None:
    with pytest.raises(error):
        RetrieverAgent(knowledge)


def test_rejects_non_request_message_type() -> None:
    request = _request().model_copy(update={"message_type": MessageType.RESULT})
    with pytest.raises(ValueError):
        RetrieverAgent(_knowledge()).handle(
            request,
            _context(HashEmbeddingProvider()),
        )


def test_rejects_non_retrieve_action() -> None:
    request = _request().model_copy(update={"action": ActionType.PLAN})
    with pytest.raises(ValueError):
        RetrieverAgent(_knowledge()).handle(
            request,
            _context(HashEmbeddingProvider()),
        )


def test_rejects_missing_plan() -> None:
    request = _request()
    request.args.pop("plan")
    with pytest.raises(ValueError):
        RetrieverAgent(_knowledge()).handle(
            request,
            _context(HashEmbeddingProvider()),
        )


@pytest.mark.parametrize("plan", [None, "plan", [], 1, True])
def test_rejects_plan_that_is_not_exact_dict(plan: object) -> None:
    request = _request()
    request.args["plan"] = plan
    with pytest.raises(TypeError):
        RetrieverAgent(_knowledge()).handle(
            request,
            _context(HashEmbeddingProvider()),
        )


def test_rejects_missing_task() -> None:
    with pytest.raises(ValueError):
        RetrieverAgent(_knowledge()).handle(
            _request(plan={"retrieval_topics": ["network"]}),
            _context(HashEmbeddingProvider()),
        )


@pytest.mark.parametrize("task", [None, 1, True, []])
def test_rejects_non_string_task(task: object) -> None:
    with pytest.raises(TypeError):
        RetrieverAgent(_knowledge()).handle(
            _request(plan={"task": task}),
            _context(HashEmbeddingProvider()),
        )


def test_rejects_empty_task() -> None:
    with pytest.raises(ValueError):
        RetrieverAgent(_knowledge()).handle(
            _request(plan={"task": ""}),
            _context(HashEmbeddingProvider()),
        )


@pytest.mark.parametrize("topics", [None, "network", ("network",), 1, True])
def test_rejects_retrieval_topics_that_are_not_list(topics: object) -> None:
    with pytest.raises(TypeError):
        RetrieverAgent(_knowledge()).handle(
            _request(plan={"task": "x", "retrieval_topics": topics}),
            _context(HashEmbeddingProvider()),
        )


def test_rejects_invalid_retrieval_topic_items() -> None:
    agent = RetrieverAgent(_knowledge())
    with pytest.raises(TypeError):
        agent.handle(
            _request(plan={"task": "x", "retrieval_topics": ["x", 1]}),
            _context(HashEmbeddingProvider()),
        )
    with pytest.raises(ValueError):
        agent.handle(
            _request(plan={"task": "x", "retrieval_topics": ["x", ""]}),
            _context(HashEmbeddingProvider()),
        )


@pytest.mark.parametrize("top_k", [True, 1.0, "3", None])
def test_top_k_requires_exact_int(top_k: object) -> None:
    with pytest.raises(TypeError):
        RetrieverAgent(_knowledge()).handle(
            _request(top_k=top_k),
            _context(HashEmbeddingProvider()),
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_positive(top_k: int) -> None:
    with pytest.raises(ValueError):
        RetrieverAgent(_knowledge()).handle(
            _request(top_k=top_k),
            _context(HashEmbeddingProvider()),
        )


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0]],
        np.zeros(3, dtype=np.float32),
        np.zeros((1, 128), dtype=np.float32),
        np.zeros((4, 64), dtype=np.float32),
        np.full((4, 128), np.nan, dtype=np.float32),
    ],
)
def test_invalid_embedding_provider_output_is_rejected(matrix: object) -> None:
    provider = BadEmbeddingProvider(matrix)
    with pytest.raises(ValueError):
        RetrieverAgent(_knowledge()).handle(_request(), _context(provider))
