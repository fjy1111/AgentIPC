from enum import Enum


PROTOCOL_VERSION = "agentipc/0.1"


class MessageType(str, Enum):
    HELLO = "HELLO"
    REGISTER = "REGISTER"
    DISCOVER = "DISCOVER"
    REQUEST = "REQUEST"
    RESULT = "RESULT"
    ACK = "ACK"
    ERROR = "ERROR"


class ActionType(str, Enum):
    PLAN = "PLAN"
    RETRIEVE = "RETRIEVE"
    EXECUTE = "EXECUTE"
    SUMMARIZE = "SUMMARIZE"
    MEMORY_QUERY = "MEMORY_QUERY"
    MEMORY_WRITE = "MEMORY_WRITE"


class MessageStatus(str, Enum):
    PENDING = "PENDING"
    OK = "OK"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"