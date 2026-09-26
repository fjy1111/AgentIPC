from datetime import datetime, timezone
from uuid import uuid4


def new_message_id() -> str:
    """Return a unique, human-readable message identifier."""
    return f"msg_{uuid4().hex}"


def new_task_id() -> str:
    """Return a unique, human-readable task identifier."""
    return f"task_{uuid4().hex}"


def new_trace_id() -> str:
    """Return a unique, human-readable trace identifier."""
    return f"trace_{uuid4().hex}"


def utc_timestamp() -> float:
    """Return the current UTC time as a Unix timestamp in seconds."""
    return datetime.now(timezone.utc).timestamp()