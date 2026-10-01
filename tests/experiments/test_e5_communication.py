from __future__ import annotations

from agentipc.evaluation.text_counter import TextCount, TextCounter
from agentipc.experiments.formal.e5_communication import run_e5


class FakeTokenCounter(TextCounter):
    def __init__(self) -> None:
        pass

    def count(self, text: str) -> TextCount:
        return TextCount(
            text_chars=len(text),
            text_tokens=max(1, len(text.encode("utf-8")) // 4),
            token_method="fake:e5-test",
        )


def test_e5_measures_all_modes_and_payload_sizes(tmp_path) -> None:
    rows, summary = run_e5(
        root=tmp_path / "e5",
        repeat=2,
        payload_sizes=(256, 2048),
        text_counter=FakeTokenCounter(),
    )
    assert len(rows) == 12
    assert summary["experiment"] == "E5"
    large = summary["payloads"]["2048"]
    assert set(large["modes"]) == {"A", "B", "C"}
    assert large["modes"]["C"]["mean_artifact_payload_bytes"] == 2048.0
    assert large["modes"]["C"]["mean_wire_bytes"] < large["modes"]["A"]["mean_wire_bytes"]
    assert large["savings_vs_A"]["C"]["wire_byte_saving_pct"] > 0.0
