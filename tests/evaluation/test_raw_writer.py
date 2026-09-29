"""Tests for raw JSONL writer and reader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentipc.evaluation.experiment import EXPERIMENT_B
from agentipc.evaluation.io import append_raw_record, read_raw_records
from agentipc.evaluation.metrics import MetricsSnapshot
from agentipc.evaluation.runner import RawRunRecord
from agentipc.runtime.context import RunMode
from agentipc.runtime.result import RunResult


def _make_record(
    *,
    task: str = "diagnose openEuler network connectivity",
    answer: str = "Network is operational.",
    success: bool = True,
    error: dict | None = None,
) -> RawRunRecord:
    """Build a valid RawRunRecord for testing."""
    metrics = MetricsSnapshot(
        message_count=8,
        text_chars=0,
        text_tokens=0,
        protocol_bytes=1234,
        state_transfer_count=1,
        state_bytes=256,
        artifact_ref_count=2,
        memory_retrieved=3,
        memory_used=2,
        memory_effective=1,
        memory_harmful=0,
        tool_call_count=2,
        repeated_tool_call_count=0,
        llm_call_count=2,
        llm_prompt_tokens=188,
        llm_completion_tokens=100,
        llm_total_tokens=288,
        llm_usage_missing_count=0,
        llm_latency_ms=123.5,
        latency_ms=130.0,
        success=success,
    )

    run_result = RunResult(
        task_id="task-test",
        success=success,
        answer=answer,
        error=error,
        metrics=metrics.model_dump(mode="json"),
        trace_path="/tmp/trace-test.jsonl",
    )

    return RawRunRecord(
        experiment=EXPERIMENT_B,
        task=task,
        task_hash="a" * 64,
        seed=42,
        use_sandbox=False,
        run_result=run_result,
    )


class TestSingleRecordRoundTrip:
    """Test basic write and read of a single record."""

    def test_single_record_round_trip(self, tmp_path: Path):
        """Write one record and read it back unchanged."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)
        records = read_raw_records(path)

        assert len(records) == 1
        assert records[0] == record

    def test_parent_directory_created_automatically(self, tmp_path: Path):
        """Parent directories are created if missing."""
        path = tmp_path / "nested" / "deep" / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)

        assert path.exists()
        assert path.parent.exists()
        records = read_raw_records(path)
        assert len(records) == 1


class TestMultipleAppend:
    """Test append semantics across multiple writes."""

    def test_multiple_append_preserves_order(self, tmp_path: Path):
        """Multiple appends preserve order."""
        path = tmp_path / "raw.jsonl"
        record1 = _make_record(task="task A")
        record2 = _make_record(task="task B")
        record3 = _make_record(task="task C")

        append_raw_record(path, record1)
        append_raw_record(path, record2)
        append_raw_record(path, record3)

        records = read_raw_records(path)
        assert len(records) == 3
        assert [r.task for r in records] == ["task A", "task B", "task C"]

    def test_append_does_not_overwrite_existing_records(self, tmp_path: Path):
        """Append mode preserves existing content."""
        path = tmp_path / "raw.jsonl"
        record1 = _make_record(task="first")
        record2 = _make_record(task="second")

        append_raw_record(path, record1)

        # Read to verify first record
        records = read_raw_records(path)
        assert len(records) == 1

        # Append second record
        append_raw_record(path, record2)

        # Both records must be present
        records = read_raw_records(path)
        assert len(records) == 2
        assert records[0].task == "first"
        assert records[1].task == "second"


class TestUnicodePreservation:
    """Test that Unicode content is preserved as literal UTF-8."""

    def test_unicode_task_preserved_as_literal_utf8(self, tmp_path: Path):
        """Chinese task text appears as literal characters, not escape sequences."""
        path = tmp_path / "raw.jsonl"
        record = _make_record(task="诊断 openEuler 网络连接")

        append_raw_record(path, record)

        raw_text = path.read_text(encoding="utf-8")
        assert "诊断 openEuler 网络连接" in raw_text
        assert "\\u" not in raw_text  # No Unicode escape sequences

    def test_unicode_answer_preserved(self, tmp_path: Path):
        """Unicode answer is preserved as literal UTF-8."""
        path = tmp_path / "raw.jsonl"
        record = _make_record(answer="网络连接正常。")

        append_raw_record(path, record)

        raw_text = path.read_text(encoding="utf-8")
        assert "网络连接正常。" in raw_text


class TestJSONFormat:
    """Test JSON formatting rules."""

    def test_one_json_object_per_physical_line(self, tmp_path: Path):
        """Each record is exactly one physical line."""
        path = tmp_path / "raw.jsonl"
        record1 = _make_record(task="first")
        record2 = _make_record(task="second")

        append_raw_record(path, record1)
        append_raw_record(path, record2)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        # Each line is valid JSON
        json.loads(lines[0])
        json.loads(lines[1])

    def test_json_keys_deterministic_and_sorted(self, tmp_path: Path):
        """JSON keys are sorted alphabetically."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)

        line = path.read_text(encoding="utf-8").splitlines()[0]
        payload = json.loads(line)

        # Re-serialize with same settings
        expected = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

        assert line == expected

    def test_compact_json_separators(self, tmp_path: Path):
        """JSON uses compact separators without extra whitespace."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)

        line = path.read_text(encoding="utf-8").splitlines()[0]

        # Verify compact format by checking structure
        # The separators are "," and ":" with no spaces
        # Parse and re-serialize to verify
        payload = json.loads(line)
        compact = json.dumps(payload, separators=(",", ":"), sort_keys=True)

        # The line should not have ": " or ", " patterns from default formatting
        # Check that our line matches compact formatting length
        assert len(line) == len(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )


class TestJSONShape:
    """Test that JSON structure contains expected top-level keys."""

    def test_top_level_keys_complete(self, tmp_path: Path):
        """JSON contains all and only the expected top-level keys."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)

        line = path.read_text(encoding="utf-8").splitlines()[0]
        payload = json.loads(line)

        expected_keys = {
            "experiment",
            "task",
            "task_hash",
            "seed",
            "use_sandbox",
            "run_result",
        }

        assert set(payload.keys()) == expected_keys


class TestEnumRoundTrip:
    """Test that Enum fields round-trip correctly."""

    def test_experiment_name_enum_preserved(self, tmp_path: Path):
        """ExperimentName enum round-trips correctly."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)
        records = read_raw_records(path)

        assert records[0].experiment.name.value == "B"
        assert records[0].experiment.mode == RunMode.STRUCTURED

    def test_run_mode_enum_preserved(self, tmp_path: Path):
        """RunMode enum round-trips correctly."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)

        line = path.read_text(encoding="utf-8").splitlines()[0]
        payload = json.loads(line)

        # Verify JSON representation
        assert payload["experiment"]["mode"] == "structured"

        # Verify deserialization
        records = read_raw_records(path)
        assert records[0].experiment.mode == RunMode.STRUCTURED


class TestMetricsPreservation:
    """Test that all metrics fields round-trip correctly."""

    def test_all_metrics_preserved(self, tmp_path: Path):
        """All metrics fields round-trip with correct values."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)
        records = read_raw_records(path)

        original_metrics = record.run_result.metrics
        restored_metrics = records[0].run_result.metrics

        assert restored_metrics["message_count"] == original_metrics["message_count"]
        assert restored_metrics["protocol_bytes"] == original_metrics["protocol_bytes"]
        assert restored_metrics["state_transfer_count"] == original_metrics["state_transfer_count"]
        assert restored_metrics["state_bytes"] == original_metrics["state_bytes"]
        assert restored_metrics["llm_call_count"] == original_metrics["llm_call_count"]
        assert restored_metrics["llm_prompt_tokens"] == original_metrics["llm_prompt_tokens"]
        assert restored_metrics["llm_completion_tokens"] == original_metrics["llm_completion_tokens"]
        assert restored_metrics["llm_total_tokens"] == original_metrics["llm_total_tokens"]
        assert restored_metrics["llm_usage_missing_count"] == original_metrics["llm_usage_missing_count"]
        assert restored_metrics["llm_latency_ms"] == original_metrics["llm_latency_ms"]
        assert restored_metrics["latency_ms"] == original_metrics["latency_ms"]
        assert restored_metrics["success"] == original_metrics["success"]


class TestFailedRecordRoundTrip:
    """Test that failed records (success=False) round-trip correctly."""

    def test_failed_record_preserves_error(self, tmp_path: Path):
        """Failed record with error dict round-trips correctly."""
        path = tmp_path / "raw.jsonl"
        error = {
            "type": "RuntimeError",
            "message": "single-run-boom",
        }
        record = _make_record(
            success=False,
            answer="",
            error=error,
        )

        append_raw_record(path, record)
        records = read_raw_records(path)

        assert len(records) == 1
        assert records[0].run_result.success is False
        assert records[0].run_result.answer == ""
        assert records[0].run_result.error == error
        assert records[0].run_result.metrics["success"] is False


class TestPathTypes:
    """Test that both str and Path are accepted."""

    def test_reader_accepts_str_path(self, tmp_path: Path):
        """read_raw_records accepts str path."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()
        append_raw_record(path, record)

        records = read_raw_records(str(path))
        assert len(records) == 1

    def test_reader_accepts_path_object(self, tmp_path: Path):
        """read_raw_records accepts Path object."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()
        append_raw_record(path, record)

        records = read_raw_records(path)
        assert len(records) == 1

    def test_writer_accepts_str_path(self, tmp_path: Path):
        """append_raw_record accepts str path."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(str(path), record)

        assert path.exists()

    def test_writer_accepts_path_object(self, tmp_path: Path):
        """append_raw_record accepts Path object."""
        path = tmp_path / "raw.jsonl"
        record = _make_record()

        append_raw_record(path, record)

        assert path.exists()


class TestInvalidInputs:
    """Test validation of invalid inputs."""

    def test_invalid_writer_path_type_rejected(self):
        """append_raw_record rejects invalid path type."""
        record = _make_record()

        with pytest.raises(TypeError, match="path must be a str or Path"):
            append_raw_record(123, record)  # type: ignore[arg-type]

    def test_invalid_reader_path_type_rejected(self):
        """read_raw_records rejects invalid path type."""
        with pytest.raises(TypeError, match="path must be a str or Path"):
            read_raw_records(123)  # type: ignore[arg-type]

    def test_wrong_record_type_rejected(self, tmp_path: Path):
        """append_raw_record rejects non-RawRunRecord."""
        path = tmp_path / "raw.jsonl"

        with pytest.raises(TypeError, match="record must be a RawRunRecord"):
            append_raw_record(path, {"task": "fake"})  # type: ignore[arg-type]

    def test_missing_file_raises_file_not_found_error(self, tmp_path: Path):
        """read_raw_records raises FileNotFoundError for missing file."""
        path = tmp_path / "does-not-exist.jsonl"

        with pytest.raises(FileNotFoundError):
            read_raw_records(path)


class TestEmptyAndBlankLines:
    """Test handling of empty files and blank lines."""

    def test_empty_file_returns_empty_list(self, tmp_path: Path):
        """Empty file returns empty list."""
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")

        records = read_raw_records(path)
        assert records == []

    def test_blank_lines_ignored(self, tmp_path: Path):
        """Blank lines are silently skipped."""
        path = tmp_path / "raw.jsonl"
        record1 = _make_record(task="first")
        record2 = _make_record(task="second")

        append_raw_record(path, record1)

        # Manually append blank lines
        with path.open("a", encoding="utf-8") as f:
            f.write("\n")
            f.write("   \n")

        append_raw_record(path, record2)

        # Should read both records, ignoring blanks
        records = read_raw_records(path)
        assert len(records) == 2
        assert records[0].task == "first"
        assert records[1].task == "second"


class TestInvalidJSONHandling:
    """Test error handling for invalid JSON."""

    def test_invalid_json_non_empty_line_propagates_error(self, tmp_path: Path):
        """Invalid JSON in non-empty line raises JSONDecodeError."""
        path = tmp_path / "bad.jsonl"
        path.write_text("{invalid json}", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            read_raw_records(path)

    def test_valid_json_invalid_record_raises_validation_error(self, tmp_path: Path):
        """Valid JSON that doesn't match RawRunRecord schema raises ValidationError."""
        path = tmp_path / "bad-schema.jsonl"
        # Valid JSON but missing required fields
        path.write_text('{"task":"foo"}', encoding="utf-8")

        with pytest.raises(ValidationError):
            read_raw_records(path)
