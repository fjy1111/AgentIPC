from __future__ import annotations

from agentipc.experiments.formal.e7_state_exchange import (
    _aggregate_e7,
    run_e7,
)
from agentipc.experiments.formal.report import render_report


def test_e7_cross_process_state_exchange_preserves_state() -> None:
    rows, summary = run_e7(
        repeat=1,
        vector_dims=(64, 256),
        use_tiktoken=False,
    )

    assert len(rows) == 4
    assert summary["passed"] is True
    assert summary["correctness_pass"] is True
    assert summary["accounting_pass"] is True

    for row in rows:
        assert row["consumer_result_match"] is True
        assert row["checksum_match"] is True
        if row["mode"] == "json_materialized":
            assert row["state_transfer_count"] == 0
            assert row["state_bytes"] == 0
        else:
            assert row["state_transfer_count"] == 1
            assert row["state_bytes"] == row["logical_state_bytes"]

    payload = summary["payloads"]["1024"]
    assert payload["vector_dim"] == 256
    assert (
        payload["modes"]["json_materialized"]["mean_wire_bytes"]
        > payload["modes"]["shm_ref"]["mean_wire_bytes"]
    )


def test_e7_aggregation_and_report_separate_wire_and_state_bytes() -> None:
    rows = [
        {
            "experiment": "E7",
            "mode": "json_materialized",
            "repeat": 1,
            "vector_dim": 1024,
            "dtype": "<f4",
            "logical_state_bytes": 4096,
            "wire_chars": 20000,
            "wire_tokens": 5000,
            "wire_bytes": 20000,
            "token_method": "tiktoken:cl100k_base",
            "state_transfer_count": 0,
            "state_bytes": 0,
            "producer_ms": 2.0,
            "pipe_send_ms": 0.2,
            "consumer_ms": 2.0,
            "validation_ms": 0.1,
            "end_to_end_ms": 5.0,
            "ipc_and_sync_ms": 1.0,
            "checksum_match": True,
            "consumer_result_match": True,
            "consumer_value": 1.0,
            "success": True,
        },
        {
            "experiment": "E7",
            "mode": "shm_ref",
            "repeat": 1,
            "vector_dim": 1024,
            "dtype": "<f4",
            "logical_state_bytes": 4096,
            "wire_chars": 500,
            "wire_tokens": 150,
            "wire_bytes": 500,
            "token_method": "tiktoken:cl100k_base",
            "state_transfer_count": 1,
            "state_bytes": 4096,
            "producer_ms": 1.0,
            "pipe_send_ms": 0.1,
            "consumer_ms": 1.0,
            "validation_ms": 0.1,
            "end_to_end_ms": 3.0,
            "ipc_and_sync_ms": 1.0,
            "checksum_match": True,
            "consumer_result_match": True,
            "consumer_value": 1.0,
            "success": True,
        },
    ]

    summary = _aggregate_e7(rows, repeat=1)
    assert summary["passed"] is True
    payload = summary["payloads"]["4096"]
    assert payload["savings"]["wire_byte_saving_pct"] == 97.5
    assert payload["savings"]["wire_token_saving_pct"] == 97.0
    assert payload["modes"]["shm_ref"]["mean_state_bytes"] == 4096.0
    assert summary["zero_copy_claim"] is False

    report = render_report(summary)
    assert "# E7 Cross-Process Non-Text State Exchange" in report
    assert "SharedMemory + StateRef" in report
    assert "does **not** claim zero-copy" in report
    assert "4096" in report
