from agentipc.protocol import (
    PROTOCOL_VERSION,
    ActionType,
    MessageStatus,
    MessageType,
)


def test_protocol_version_is_fixed_mvp_version() -> None:
    assert PROTOCOL_VERSION == "agentipc/0.1"


def test_message_type_members_match_contract_exactly() -> None:
    expected = {
        "HELLO",
        "REGISTER",
        "DISCOVER",
        "REQUEST",
        "RESULT",
        "ACK",
        "ERROR",
    }

    assert {member.name for member in MessageType} == expected
    assert {member.value for member in MessageType} == expected


def test_action_type_members_match_contract_exactly() -> None:
    expected = {
        "PLAN",
        "RETRIEVE",
        "EXECUTE",
        "SUMMARIZE",
        "MEMORY_QUERY",
        "MEMORY_WRITE",
    }

    assert {member.name for member in ActionType} == expected
    assert {member.value for member in ActionType} == expected


def test_message_status_members_match_contract_exactly() -> None:
    expected = {
        "PENDING",
        "OK",
        "ERROR",
        "TIMEOUT",
        "SKIPPED",
    }

    assert {member.name for member in MessageStatus} == expected
    assert {member.value for member in MessageStatus} == expected


def test_protocol_enums_are_string_compatible() -> None:
    assert isinstance(MessageType.HELLO, str)
    assert isinstance(ActionType.PLAN, str)
    assert isinstance(MessageStatus.PENDING, str)


def test_protocol_enum_values_match_names() -> None:
    for enum_type in (MessageType, ActionType, MessageStatus):
        assert all(member.value == member.name for member in enum_type)