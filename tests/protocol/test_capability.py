import pytest
from pydantic import ValidationError

from agentipc.protocol.capability import AgentCapability
from agentipc.protocol.enums import PROTOCOL_VERSION


def test_minimal_capability_uses_expected_defaults() -> None:
    capability = AgentCapability(agent_id="planner")

    assert capability.agent_id == "planner"
    assert capability.capabilities == []
    assert capability.protocol_versions == [PROTOCOL_VERSION]
    assert capability.metadata == {}


def test_agent_id_must_be_non_empty() -> None:
    with pytest.raises(ValidationError):
        AgentCapability(agent_id="")


def test_mutable_defaults_are_independent_between_instances() -> None:
    first = AgentCapability(agent_id="planner")
    second = AgentCapability(agent_id="retriever")

    first.capabilities.append("planning")
    first.protocol_versions.append("agentipc/test")
    first.metadata["role"] = "planner"

    assert second.capabilities == []
    assert second.protocol_versions == [PROTOCOL_VERSION]
    assert second.metadata == {}


def test_capabilities_reject_empty_string() -> None:
    with pytest.raises(ValidationError):
        AgentCapability(agent_id="planner", capabilities=[""])


def test_protocol_versions_reject_empty_string() -> None:
    with pytest.raises(ValidationError):
        AgentCapability(agent_id="planner", protocol_versions=[""])


def test_extra_field_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentCapability(agent_id="planner", unexpected=True)


def test_model_dump_and_validate_round_trip() -> None:
    capability = AgentCapability(
        agent_id="planner",
        capabilities=["planning", "summarization"],
        protocol_versions=[PROTOCOL_VERSION],
        metadata={"role": "planner"},
    )

    restored = AgentCapability.model_validate(capability.model_dump())

    assert restored == capability