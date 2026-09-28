import pytest

from agentipc.agents.base import BaseAgent
from agentipc.runtime.agent_registry import AgentRegistry


class FakeAgent(BaseAgent):
    capabilities = ["fake"]

    def __init__(self, agent_id: object = "fake") -> None:
        self.agent_id = agent_id  # type: ignore[assignment]

    def handle(self, envelope, ctx):
        return envelope


def test_register_and_get_returns_identical_agent() -> None:
    registry = AgentRegistry()
    agent = FakeAgent("fake")

    registry.register(agent)

    assert registry.get("fake") is agent


def test_multiple_agent_ids_can_coexist() -> None:
    registry = AgentRegistry()
    planner = FakeAgent("planner-fake")
    retriever = FakeAgent("retriever-fake")

    registry.register(planner)
    registry.register(retriever)

    assert registry.get("planner-fake") is planner
    assert registry.get("retriever-fake") is retriever


def test_duplicate_registration_raises_and_preserves_first_agent() -> None:
    registry = AgentRegistry()
    first = FakeAgent("same")
    second = FakeAgent("same")
    registry.register(first)

    with pytest.raises(ValueError):
        registry.register(second)

    assert registry.get("same") is first


def test_registering_same_instance_twice_is_duplicate() -> None:
    registry = AgentRegistry()
    agent = FakeAgent("same-instance")
    registry.register(agent)

    with pytest.raises(ValueError):
        registry.register(agent)

    assert registry.get("same-instance") is agent


def test_unknown_agent_raises_key_error() -> None:
    registry = AgentRegistry()

    with pytest.raises(KeyError):
        registry.get("missing")


@pytest.mark.parametrize("invalid_agent", [object(), None, "agent"])
def test_register_rejects_non_base_agent(invalid_agent: object) -> None:
    registry = AgentRegistry()

    with pytest.raises(TypeError):
        registry.register(invalid_agent)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("agent_id", "expected_exception"),
    [
        ("", ValueError),
        (123, TypeError),
    ],
)
def test_register_validates_agent_id(agent_id: object, expected_exception: type[Exception]) -> None:
    registry = AgentRegistry()

    with pytest.raises(expected_exception):
        registry.register(FakeAgent(agent_id))


@pytest.mark.parametrize(
    ("agent_id", "expected_exception"),
    [
        ("", ValueError),
        (123, TypeError),
    ],
)
def test_get_validates_agent_id(agent_id: object, expected_exception: type[Exception]) -> None:
    registry = AgentRegistry()

    with pytest.raises(expected_exception):
        registry.get(agent_id)  # type: ignore[arg-type]
