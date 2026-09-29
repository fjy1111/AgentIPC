"""Tests for timestamped result directory creation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from agentipc.evaluation.io import create_result_dir


class TestResultsRootHandling:
    """Test results_root path handling."""

    def test_str_root_accepted(self, tmp_path: Path):
        """create_result_dir accepts str root."""
        root = str(tmp_path / "results")
        result_dir = create_result_dir(root, "test")

        assert result_dir.exists()
        assert result_dir.parent == Path(root)

    def test_path_root_accepted(self, tmp_path: Path):
        """create_result_dir accepts Path root."""
        root = tmp_path / "results"
        result_dir = create_result_dir(root, "test")

        assert result_dir.exists()
        assert result_dir.parent == root

    def test_root_created_automatically(self, tmp_path: Path):
        """results_root is created if it does not exist."""
        root = tmp_path / "nested" / "deep" / "results"
        assert not root.exists()

        result_dir = create_result_dir(root, "test")

        assert root.exists()
        assert result_dir.exists()
        assert result_dir.parent == root

    def test_invalid_root_type_rejected(self):
        """create_result_dir rejects invalid root type."""
        with pytest.raises(TypeError, match="results_root must be a str or Path"):
            create_result_dir(123, "test")  # type: ignore[arg-type]


class TestSuiteValidation:
    """Test suite identifier validation."""

    def test_valid_suite_accepted(self, tmp_path: Path):
        """Valid suite identifiers are accepted."""
        valid_suites = [
            "abcd",
            "test_suite",
            "suite-123",
            "suite.v1",
            "A1B2C3",
            "suite_2024_01_15",
        ]

        for suite in valid_suites:
            result_dir = create_result_dir(tmp_path, suite)
            assert result_dir.exists()
            assert suite in result_dir.name

    def test_suite_must_be_exact_str(self, tmp_path: Path):
        """suite must be an exact str, not a subclass."""
        with pytest.raises(TypeError, match="suite must be an exact str"):
            create_result_dir(tmp_path, 123)  # type: ignore[arg-type]

    def test_empty_suite_rejected(self, tmp_path: Path):
        """Empty suite string is rejected."""
        with pytest.raises(ValueError, match="suite must be a non-empty str"):
            create_result_dir(tmp_path, "")

    def test_dot_rejected(self, tmp_path: Path):
        """Suite '.' is rejected."""
        with pytest.raises(ValueError, match=r"suite must not be '\.' or '\.\.'"):
            create_result_dir(tmp_path, ".")

    def test_dot_dot_rejected(self, tmp_path: Path):
        """Suite '..' is rejected."""
        with pytest.raises(ValueError, match=r"suite must not be '\.' or '\.\.'"):
            create_result_dir(tmp_path, "..")

    def test_forward_slash_rejected(self, tmp_path: Path):
        """Suite containing '/' is rejected."""
        with pytest.raises(ValueError, match="suite must not contain path separators"):
            create_result_dir(tmp_path, "a/b")

    def test_backslash_rejected(self, tmp_path: Path):
        """Suite containing backslash is rejected."""
        with pytest.raises(ValueError, match="suite must not contain path separators"):
            create_result_dir(tmp_path, "a\\b")

    def test_path_traversal_rejected(self, tmp_path: Path):
        """Suite containing path traversal is rejected."""
        with pytest.raises(ValueError, match="suite must not contain path separators"):
            create_result_dir(tmp_path, "../evil")

    def test_suite_must_start_with_alphanumeric(self, tmp_path: Path):
        """Suite must start with alphanumeric character."""
        with pytest.raises(ValueError, match="suite must start with an alphanumeric character"):
            create_result_dir(tmp_path, "_suite")

        with pytest.raises(ValueError, match="suite must start with an alphanumeric character"):
            create_result_dir(tmp_path, "-suite")

    def test_invalid_characters_rejected(self, tmp_path: Path):
        """Suite with invalid characters is rejected."""
        with pytest.raises(ValueError, match="suite must contain only alphanumeric"):
            create_result_dir(tmp_path, "suite@test")

        with pytest.raises(ValueError, match="suite must contain only alphanumeric"):
            create_result_dir(tmp_path, "suite test")  # space


class TestTimestampHandling:
    """Test timestamp parameter and formatting."""

    def test_default_timestamp_uses_current_utc(self, tmp_path: Path):
        """Default timestamp (None) uses current UTC time."""
        result_dir = create_result_dir(tmp_path, "test")

        # Directory name should start with a timestamp in expected format
        name = result_dir.name
        timestamp_part = name.split("-")[0]

        # Should be YYYYMMDDTHHMMSSffffffZ format
        assert timestamp_part.endswith("Z")
        assert "T" in timestamp_part
        assert len(timestamp_part) == 22  # YYYYMMDDTHHMMSS (14) + ffffff (6) + Z (1) + T (1)

    def test_explicit_timestamp_deterministic(self, tmp_path: Path):
        """Explicit timestamp produces deterministic directory name."""
        ts = datetime(2026, 9, 29, 15, 30, 45, 123456, tzinfo=timezone.utc)
        result_dir = create_result_dir(tmp_path, "test", timestamp=ts)

        expected_prefix = "20260929T153045123456Z-test"
        assert result_dir.name == expected_prefix

    def test_timezone_aware_timestamp_converted_to_utc(self, tmp_path: Path):
        """Timezone-aware timestamp is converted to UTC."""
        # Create a timestamp in UTC+8
        from datetime import timedelta
        tz_plus8 = timezone(timedelta(hours=8))
        ts_local = datetime(2026, 9, 29, 23, 30, 0, 0, tzinfo=tz_plus8)

        # This should be converted to 2026-09-29 15:30:00 UTC
        result_dir = create_result_dir(tmp_path, "test", timestamp=ts_local)

        expected_prefix = "20260929T153000000000Z-test"
        assert result_dir.name == expected_prefix

    def test_naive_timestamp_rejected(self, tmp_path: Path):
        """Naive datetime (no timezone) is rejected."""
        naive_ts = datetime(2026, 9, 29, 15, 30, 0)

        with pytest.raises(ValueError, match="timestamp must be timezone-aware, not naive"):
            create_result_dir(tmp_path, "test", timestamp=naive_ts)

    def test_invalid_timestamp_type_rejected(self, tmp_path: Path):
        """Invalid timestamp type is rejected."""
        with pytest.raises(TypeError, match="timestamp must be a datetime or None"):
            create_result_dir(tmp_path, "test", timestamp="2026-09-29")  # type: ignore[arg-type]


class TestDirectoryNameFormat:
    """Test directory name format and structure."""

    def test_directory_name_format(self, tmp_path: Path):
        """Directory name follows YYYYMMDDTHHMMSSffffffZ-suite format."""
        ts = datetime(2026, 1, 5, 8, 15, 30, 500000, tzinfo=timezone.utc)
        result_dir = create_result_dir(tmp_path, "abcd", timestamp=ts)

        expected = "20260105T081530500000Z-abcd"
        assert result_dir.name == expected

    def test_timestamp_and_suite_separated_by_hyphen(self, tmp_path: Path):
        """Timestamp and suite are separated by exactly one hyphen."""
        ts = datetime(2026, 9, 29, 12, 0, 0, 0, tzinfo=timezone.utc)
        result_dir = create_result_dir(tmp_path, "suite", timestamp=ts)

        parts = result_dir.name.split("-")
        assert len(parts) == 2
        assert parts[0] == "20260929T120000000000Z"
        assert parts[1] == "suite"


class TestExistenceHandling:
    """Test behavior when target directory already exists."""

    def test_duplicate_timestamp_and_suite_raises_file_exists_error(self, tmp_path: Path):
        """Creating the same directory twice raises FileExistsError."""
        ts = datetime(2026, 9, 29, 15, 0, 0, 0, tzinfo=timezone.utc)

        # First creation succeeds
        result_dir = create_result_dir(tmp_path, "test", timestamp=ts)
        assert result_dir.exists()

        # Second creation with same timestamp and suite fails
        with pytest.raises(FileExistsError):
            create_result_dir(tmp_path, "test", timestamp=ts)

    def test_existing_directory_not_overwritten(self, tmp_path: Path):
        """Existing directory contents are preserved on conflict."""
        ts = datetime(2026, 9, 29, 15, 0, 0, 0, tzinfo=timezone.utc)

        # Create directory and add a file
        result_dir = create_result_dir(tmp_path, "test", timestamp=ts)
        marker_file = result_dir / "important.txt"
        marker_file.write_text("do not delete", encoding="utf-8")

        # Attempt to create again should fail
        with pytest.raises(FileExistsError):
            create_result_dir(tmp_path, "test", timestamp=ts)

        # Original file must still exist
        assert marker_file.exists()
        assert marker_file.read_text(encoding="utf-8") == "do not delete"

    def test_different_suite_same_timestamp_allowed(self, tmp_path: Path):
        """Different suite with same timestamp creates separate directory."""
        ts = datetime(2026, 9, 29, 15, 0, 0, 0, tzinfo=timezone.utc)

        dir1 = create_result_dir(tmp_path, "suite1", timestamp=ts)
        dir2 = create_result_dir(tmp_path, "suite2", timestamp=ts)

        assert dir1.exists()
        assert dir2.exists()
        assert dir1 != dir2


class TestReturnValue:
    """Test function return value."""

    def test_returns_path_to_created_directory(self, tmp_path: Path):
        """create_result_dir returns Path to the created directory."""
        result_dir = create_result_dir(tmp_path, "test")

        assert isinstance(result_dir, Path)
        assert result_dir.exists()
        assert result_dir.is_dir()

    def test_returned_path_is_absolute(self, tmp_path: Path):
        """Returned path is absolute."""
        result_dir = create_result_dir(tmp_path, "test")

        assert result_dir.is_absolute()

    def test_returned_path_is_child_of_results_root(self, tmp_path: Path):
        """Returned path is direct child of results_root."""
        root = tmp_path / "results"
        result_dir = create_result_dir(root, "test")

        assert result_dir.parent == root
