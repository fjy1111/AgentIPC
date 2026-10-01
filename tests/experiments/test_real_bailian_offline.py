from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pytest

from agentipc.experiments.real_bailian.config import RealBailianConfig
from agentipc.experiments.real_bailian.providers import (
    RecordingEmbeddingProvider,
    RecordingLLMProvider,
    build_provider_usage,
)
from agentipc.experiments.real_bailian.artifacts import (
    audit_result_secrets,
    create_result_dir,
)
from agentipc.experiments.real_bailian.environment import capture_environment_manifest
from agentipc.providers.base import LLMResponse


class _FakeLLM:
    def complete(self, messages, *, temperature=0.0):
        return LLMResponse(
            text="probe answer that must not be recorded",
            prompt_tokens=7,
            completion_tokens=3,
            latency_ms=1.25,
            raw={
                "provider": "fake",
                "model": "fake-model",
                "request_id": "req-1",
                "finish_reason": "stop",
            },
        )


class _FakeEmbedding:
    @property
    def dim(self):
        return 4

    def embed(self, texts):
        return np.ones((len(texts), 4), dtype=np.float32)


def _config() -> RealBailianConfig:
    return RealBailianConfig(
        api_key="sk-unit-test-secret",
        base_url="https://secret-workspace.invalid/v1",
        region="singapore",
    )


def test_recording_providers_capture_usage_not_payload():
    llm = RecordingLLMProvider(_FakeLLM(), model="fake-model")
    embedding = RecordingEmbeddingProvider(_FakeEmbedding(), model="fake-embed")

    response = llm.complete([{"role": "user", "content": "private prompt"}])
    matrix = embedding.embed(["private embedding input", "another input"])
    usage = build_provider_usage(llm, embedding).model_dump(mode="json")
    serialized = json.dumps(usage)

    assert response.text == "probe answer that must not be recorded"
    assert matrix.shape == (2, 4)
    assert usage["llm_prompt_tokens"] == 7
    assert usage["llm_completion_tokens"] == 3
    assert usage["llm_total_tokens"] == 10
    assert usage["embedding_input_count"] == 2
    assert "private prompt" not in serialized
    assert "probe answer that must not be recorded" not in serialized
    assert "private embedding input" not in serialized


def test_result_directory_never_overwrites(tmp_path):
    now = datetime(2026, 10, 1, 1, 2, 3, tzinfo=timezone.utc)
    first = create_result_dir(tmp_path, phase="calibration", now=now)
    assert first.exists()

    with pytest.raises(FileExistsError):
        create_result_dir(tmp_path, phase="calibration", now=now)


def test_environment_manifest_excludes_api_key_and_base_url(tmp_path):
    config = _config()
    manifest = capture_environment_manifest(
        config=config,
        repo_root=tmp_path,
        started_at=datetime.now(timezone.utc),
        repeat=1,
        task_count=4,
    )
    serialized = json.dumps(manifest)

    assert config.api_key not in serialized
    assert config.base_url not in serialized
    assert manifest["api_region"] == "singapore"
    assert manifest["state_transport"] == "shm"


def test_secret_audit_detects_exact_secret_without_echoing_it(tmp_path):
    config = _config()
    for name in (
        "environment.json",
        "provider_usage.json",
        "raw.jsonl",
        "summary.json",
        "report.md",
    ):
        (tmp_path / name).write_text("safe", encoding="utf-8")

    assert audit_result_secrets(tmp_path, config=config)["passed"] is True

    (tmp_path / "raw.jsonl").write_text(config.api_key, encoding="utf-8")
    audit = audit_result_secrets(tmp_path, config=config)
    assert audit["passed"] is False
    assert audit["finding_files"] == ["raw.jsonl"]
    assert config.api_key not in json.dumps(audit)


def test_full_calibration_is_offline_with_fake_providers(tmp_path, monkeypatch):
    """Exercise the real harness without network when the full checkout is present."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "src/agentipc/runtime/orchestrator.py").exists():
        pytest.skip("full AgentIPC checkout is not present in this test workspace")

    from agentipc.experiments.real_bailian import runner as runner_module

    class FakeRealLLM:
        def complete(self, messages, *, temperature=0.0):
            return LLMResponse(
                text="nmcli device show",
                prompt_tokens=5,
                completion_tokens=2,
                latency_ms=1.0,
                raw={
                    "provider": "fake_real",
                    "model": "fake-qwen",
                    "request_id": "fake-request",
                    "finish_reason": "stop",
                },
            )

    class FakeRealEmbedding:
        @property
        def dim(self):
            return 1024

        def embed(self, texts):
            matrix = np.ones((len(texts), 1024), dtype=np.float32)
            matrix /= np.float32(np.sqrt(1024.0))
            return matrix

    llm = RecordingLLMProvider(FakeRealLLM(), model="fake-qwen")
    embedding = RecordingEmbeddingProvider(
        FakeRealEmbedding(),
        model="fake-qwen-embedding",
    )
    monkeypatch.setattr(
        runner_module,
        "build_recording_provider_bundle",
        lambda config: (llm, embedding),
    )

    result_dir, summary = runner_module.run_calibration(
        config=_config(),
        repo_root=repo_root,
        results_root=tmp_path,
    )

    assert summary["pass"] is True
    assert summary["provider_probe"]["passed"] is True
    assert summary["embedding_probe"]["passed"] is True
    assert summary["shm_probe"]["transport"] == "shm"
    assert summary["knowledge"]["passed"] is True
    assert summary["codeact"]["passed"] is True
    assert summary["codeact"]["rounds"][1]["metrics"]["tool_call_count"] == 0
    assert summary["secret_audit"]["passed"] is True
    assert (result_dir / "report.md").exists()
