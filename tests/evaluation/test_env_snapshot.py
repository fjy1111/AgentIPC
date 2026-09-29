"""Tests for environment snapshot capture."""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest

from agentipc.config import AgentIPCConfig
from agentipc.evaluation.env import EnvironmentSnapshot, capture_environment_snapshot


class TestEnvironmentSnapshotModel:
    """Test EnvironmentSnapshot data model."""

    def test_json_serializable(self):
        """EnvironmentSnapshot can be serialized to JSON."""
        snapshot = EnvironmentSnapshot(
            os_name="posix",
            platform="Linux-5.10.0-x86_64",
            python_version="3.10.12",
            python_implementation="CPython",
            llm_provider="mock",
            embedding_provider="hash",
            token_method="tiktoken:cl100k_base",
            dependencies={
                "pydantic": "2.5.0",
                "numpy": "1.24.3",
                "psutil": "5.9.5",
                "PyYAML": "6.0.1",
                "openai": None,
                "tiktoken": "0.7.0",
                "sentence-transformers": None,
            },
        )

        # Must serialize to JSON without error
        data = snapshot.model_dump(mode="json")
        json_str = json.dumps(data)

        # Must deserialize back to equivalent model
        parsed = json.loads(json_str)
        restored = EnvironmentSnapshot.model_validate(parsed)

        assert restored == snapshot

    def test_extra_fields_forbidden(self):
        """EnvironmentSnapshot rejects extra fields."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            EnvironmentSnapshot(
                os_name="posix",
                platform="Linux",
                python_version="3.10.0",
                python_implementation="CPython",
                llm_provider="mock",
                embedding_provider="hash",
                token_method="unavailable",
                dependencies={},
                extra_field="not allowed",  # type: ignore[call-arg]
            )


class TestCaptureEnvironmentSnapshot:
    """Test capture_environment_snapshot function."""

    def test_config_type_validation(self):
        """capture_environment_snapshot rejects non-AgentIPCConfig."""
        with pytest.raises(TypeError, match="config must be an AgentIPCConfig"):
            capture_environment_snapshot({"llm_provider": "mock"})  # type: ignore[arg-type]

    def test_os_and_platform_fields_populated(self):
        """OS and platform fields are non-empty."""
        config = AgentIPCConfig()
        snapshot = capture_environment_snapshot(config)

        assert snapshot.os_name != ""
        assert snapshot.platform != ""
        assert snapshot.python_version != ""
        assert snapshot.python_implementation != ""

    def test_provider_fields_preserved_from_config(self):
        """Provider fields match config values."""
        config = AgentIPCConfig(
            llm_provider="openai",
            embedding_provider="sentence-transformer",
        )
        snapshot = capture_environment_snapshot(config)

        assert snapshot.llm_provider == "openai"
        assert snapshot.embedding_provider == "sentence-transformer"

    def test_default_provider_fields_preserved(self):
        """Default provider values are preserved."""
        config = AgentIPCConfig()
        snapshot = capture_environment_snapshot(config)

        assert snapshot.llm_provider == "mock"
        assert snapshot.embedding_provider == "hash"

    def test_dependencies_keys_complete(self):
        """All expected dependency keys are present."""
        config = AgentIPCConfig()
        snapshot = capture_environment_snapshot(config)

        expected_keys = {
            "pydantic",
            "numpy",
            "psutil",
            "PyYAML",
            "openai",
            "tiktoken",
            "sentence-transformers",
        }

        assert set(snapshot.dependencies.keys()) == expected_keys

    def test_core_dependencies_have_versions(self):
        """Core dependencies (pydantic, numpy, psutil, PyYAML) have versions."""
        config = AgentIPCConfig()
        snapshot = capture_environment_snapshot(config)

        # These are required dependencies and must be installed
        assert snapshot.dependencies["pydantic"] is not None
        assert snapshot.dependencies["numpy"] is not None
        assert snapshot.dependencies["psutil"] is not None
        assert snapshot.dependencies["PyYAML"] is not None

        # Versions must be non-empty strings
        assert isinstance(snapshot.dependencies["pydantic"], str)
        assert snapshot.dependencies["pydantic"] != ""

    def test_missing_optional_dependency_returns_none(self):
        """Missing optional dependencies return None, not an error."""
        config = AgentIPCConfig()

        with patch("agentipc.evaluation.env._distribution_version") as mock_version:
            # Mock all dependencies as missing
            mock_version.return_value = None

            snapshot = capture_environment_snapshot(config)

            # All dependencies should be None
            for dep_name, dep_version in snapshot.dependencies.items():
                assert dep_version is None, f"{dep_name} should be None"

    def test_installed_dependency_returns_version_string(self):
        """Installed dependencies return version strings."""
        config = AgentIPCConfig()

        with patch("agentipc.evaluation.env._distribution_version") as mock_version:
            # Mock pydantic as installed with a specific version
            def version_mock(name: str) -> str | None:
                if name == "pydantic":
                    return "2.5.0"
                return None

            mock_version.side_effect = version_mock

            snapshot = capture_environment_snapshot(config)

            assert snapshot.dependencies["pydantic"] == "2.5.0"
            assert snapshot.dependencies["numpy"] is None

    def test_token_method_tiktoken_available(self):
        """Token method reports tiktoken when available."""
        config = AgentIPCConfig()

        # Mock TextCounter to simulate tiktoken available
        mock_counter = Mock()
        mock_count_result = Mock()
        mock_count_result.token_method = "tiktoken:cl100k_base"
        mock_counter.count.return_value = mock_count_result

        with patch("agentipc.evaluation.env.TextCounter", return_value=mock_counter):
            snapshot = capture_environment_snapshot(config)

            assert snapshot.token_method == "tiktoken:cl100k_base"

    def test_token_method_unavailable(self):
        """Token method reports unavailable when tiktoken is missing."""
        config = AgentIPCConfig()

        # Mock TextCounter to simulate tiktoken unavailable
        mock_counter = Mock()
        mock_count_result = Mock()
        mock_count_result.token_method = "unavailable"
        mock_counter.count.return_value = mock_count_result

        with patch("agentipc.evaluation.env.TextCounter", return_value=mock_counter):
            snapshot = capture_environment_snapshot(config)

            assert snapshot.token_method == "unavailable"


class TestSecurityConstraints:
    """Test that sensitive data is never captured."""

    def test_no_api_key_in_snapshot(self):
        """Snapshot does not contain API keys."""
        config = AgentIPCConfig()
        snapshot = capture_environment_snapshot(config)

        # Serialize to check content
        data = snapshot.model_dump(mode="json")
        json_str = json.dumps(data)

        # API key terms must not appear
        assert "api_key" not in json_str.lower()
        assert "apikey" not in json_str.lower()
        assert "token" not in json_str.lower() or "token_method" in json_str.lower()

    def test_no_base_url_in_snapshot(self):
        """Snapshot does not contain base URLs."""
        config = AgentIPCConfig()
        snapshot = capture_environment_snapshot(config)

        data = snapshot.model_dump(mode="json")
        json_str = json.dumps(data)

        # Base URL must not appear
        assert "base_url" not in json_str.lower()
        assert "http://" not in json_str.lower()
        assert "https://" not in json_str.lower()

    def test_no_environment_dump_in_snapshot(self):
        """Snapshot does not dump os.environ."""
        config = AgentIPCConfig()
        snapshot = capture_environment_snapshot(config)

        # Snapshot must not have an 'environ' or 'env_vars' field
        data = snapshot.model_dump(mode="json")

        assert "environ" not in data
        assert "env_vars" not in data
        assert "environment" not in data

    def test_provider_mode_recorded_not_credentials(self):
        """Provider mode is recorded, but credentials are not."""
        config = AgentIPCConfig(llm_provider="openai")
        snapshot = capture_environment_snapshot(config)

        # Provider mode is present
        assert snapshot.llm_provider == "openai"

        # But no credential fields exist
        data = snapshot.model_dump(mode="json")
        assert "api_key" not in str(data).lower()


class TestRoundTripSerialization:
    """Test that snapshots round-trip through JSON."""

    def test_full_round_trip(self):
        """Snapshot survives JSON round-trip unchanged."""
        config = AgentIPCConfig()
        original = capture_environment_snapshot(config)

        # Serialize to JSON
        json_data = original.model_dump(mode="json")
        json_str = json.dumps(json_data)

        # Deserialize from JSON
        parsed = json.loads(json_str)
        restored = EnvironmentSnapshot.model_validate(parsed)

        # Must be equal
        assert restored == original
