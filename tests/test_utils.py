import time

import pytest

from agentipc.utils import new_message_id, new_task_id, new_trace_id, utc_timestamp


@pytest.mark.parametrize(
    ("factory", "prefix"),
    [
        (new_message_id, "msg_"),
        (new_task_id, "task_"),
        (new_trace_id, "trace_"),
    ],
)
def test_id_helpers_return_unique_prefixed_ids(factory, prefix: str) -> None:
    first = factory()
    second = factory()

    assert first.startswith(prefix)
    assert second.startswith(prefix)
    assert first != second


def test_utc_timestamp_returns_current_unix_time() -> None:
    timestamp = utc_timestamp()

    assert isinstance(timestamp, float)
    assert timestamp > 0
    assert abs(timestamp - time.time()) < 1.0