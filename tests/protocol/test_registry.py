from agentipc.protocol.capability import AgentCapability
from agentipc.protocol.registry import CapabilityRegistry


def make_capability(
    agent_id: str,
    capabilities: list[str],
) -> AgentCapability:
    return AgentCapability(
        agent_id=agent_id,
        capabilities=capabilities,
    )


def test_register_and_get() -> None:
    registry = CapabilityRegistry()
    capability = make_capability("planner", ["planning"])

    registry.register(capability)

    assert registry.get("planner") == capability


def test_get_unknown_agent_returns_none() -> None:
    registry = CapabilityRegistry()

    assert registry.get("missing") is None


def test_duplicate_registration_fully_replaces_previous_record() -> None:
    registry = CapabilityRegistry()
    first = AgentCapability(
        agent_id="planner",
        capabilities=["planning"],
        metadata={"generation": 1},
    )
    second = AgentCapability(
        agent_id="planner",
        capabilities=["planning", "summarization"],
        metadata={"generation": 2},
    )

    registry.register(first)
    registry.register(second)

    assert registry.get("planner") == second
    assert registry.get("planner") != first


def test_discover_matches_exact_capability_and_sorts_by_agent_id() -> None:
    registry = CapabilityRegistry()
    registry.register(make_capability("planner", ["planning"]))
    registry.register(make_capability("retriever-b", ["retrieval"]))
    registry.register(
        make_capability(
            "retriever-a",
            ["retrieval", "search"],
        )
    )

    matches = registry.discover("retrieval")

    assert [capability.agent_id for capability in matches] == [
        "retriever-a",
        "retriever-b",
    ]


def test_discover_does_not_use_substring_matching() -> None:
    registry = CapabilityRegistry()
    registry.register(make_capability("retriever", ["retrieval"]))

    assert registry.discover("retrieve") == []


def test_discover_returns_empty_list_when_no_agent_matches() -> None:
    registry = CapabilityRegistry()
    registry.register(make_capability("planner", ["planning"]))

    assert registry.discover("retrieval") == []


def test_supports_uses_exact_membership() -> None:
    registry = CapabilityRegistry()
    registry.register(
        make_capability(
            "retriever",
            ["retrieval", "search"],
        )
    )

    assert registry.supports("retriever", "retrieval") is True
    assert registry.supports("retriever", "retrieve") is False
    assert registry.supports("retriever", "planning") is False


def test_supports_unknown_agent_returns_false() -> None:
    registry = CapabilityRegistry()

    assert registry.supports("missing", "retrieval") is False