from __future__ import annotations

import pytest

from agentipc.experiments.real_bailian import cli
from agentipc.experiments.real_bailian.config import (
    RealBailianConfig,
    RealBailianConfigError,
    load_real_bailian_config,
)


class _ExplodingEnvironment(dict[str, str]):
    def get(self, key, default=None):
        raise AssertionError("environment must not be read before confirmation")


def test_missing_confirm_flag_reads_no_environment(capsys):
    result = cli.main(["calibration"], environ=_ExplodingEnvironment())

    assert result != 0
    assert capsys.readouterr().out.strip() == (
        "Real API execution requires --confirm-real-api"
    )


def test_missing_api_key_is_clean_failure():
    with pytest.raises(RealBailianConfigError, match="DASHSCOPE_API_KEY"):
        load_real_bailian_config(
            {
                "AGENTIPC_BAILIAN_BASE_URL": "https://example.invalid/v1",
                "AGENTIPC_BAILIAN_REGION": "singapore",
            }
        )


def test_missing_base_url_is_clean_failure():
    with pytest.raises(RealBailianConfigError, match="AGENTIPC_BAILIAN_BASE_URL"):
        load_real_bailian_config(
            {
                "DASHSCOPE_API_KEY": "secret-key",
                "AGENTIPC_BAILIAN_REGION": "singapore",
            }
        )


def test_config_repr_and_public_fields_never_serialize_secret():
    secret = "sk-super-secret"
    base_url = "https://workspace-secret.example.invalid/v1"
    config = RealBailianConfig(
        api_key=secret,
        base_url=base_url,
        region="singapore",
    )

    rendered = repr(config)
    public = repr(config.public_fields())
    assert secret not in rendered
    assert base_url not in rendered
    assert secret not in public
    assert base_url not in public
    assert config.public_fields()["api_region"] == "singapore"


def test_region_rejects_url_or_workspace_identifier():
    with pytest.raises(ValueError, match="short label"):
        RealBailianConfig(
            api_key="secret",
            base_url="https://example.invalid/v1",
            region="https://example.invalid/workspaces/123",
        )
