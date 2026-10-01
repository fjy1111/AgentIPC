from agentipc.experiments.real_bailian.report import render_calibration_report


def test_report_distinguishes_strict_validated_and_token_semantics():
    report = render_calibration_report(
        {
            "passed": True,
            "provider_probe": {"passed": True},
            "embedding_probe": {"passed": True},
            "shm_probe": {"passed": True},
            "knowledge": {"passed": True, "rounds": []},
            "codeact": {"passed": True, "rounds": []},
            "provider_usage": {},
            "secret_audit": {"passed": True, "files_scanned": 5},
        }
    )

    assert "runtime_memory_effective_strict" in report
    assert "validated_memory_effective" in report
    assert "exact historical-vs-final answer string equality" in report
    assert "communication-side token estimate" in report
    assert "not Bailian billing usage" in report
