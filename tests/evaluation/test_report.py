"""Tests for markdown report renderer."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentipc.evaluation.derived import DerivedMetrics
from agentipc.evaluation.experiment import ExperimentName
from agentipc.evaluation.io import BenchmarkSummary, ExperimentSummary
from agentipc.evaluation.report import render_markdown_report, write_markdown_report
from agentipc.evaluation.stats import AggregateStats


def _build_minimal_summary() -> BenchmarkSummary:
    """Build a minimal BenchmarkSummary fixture."""
    return BenchmarkSummary(
        total_records=4,
        task_count=1,
        seed_count=1,
        experiments={
            "A": ExperimentSummary(
                experiment=ExperimentName.A,
                run_count=1,
                success_count=1,
                failure_count=0,
                success_rate=1.0,
                metrics={
                    "message_count": AggregateStats(count=1, mean=8.0, std=0.0, min=8.0, max=8.0),
                    "text_chars": AggregateStats(count=1, mean=5000.0, std=0.0, min=5000.0, max=5000.0),
                    "text_tokens": AggregateStats(count=1, mean=1000.0, std=0.0, min=1000.0, max=1000.0),
                    "protocol_bytes": AggregateStats(count=1, mean=2000.0, std=0.0, min=2000.0, max=2000.0),
                    "state_transfer_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "state_bytes": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "artifact_ref_count": AggregateStats(count=1, mean=1.0, std=0.0, min=1.0, max=1.0),
                    "memory_retrieved": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_used": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_effective": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_harmful": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "tool_call_count": AggregateStats(count=1, mean=3.0, std=0.0, min=3.0, max=3.0),
                    "repeated_tool_call_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "llm_call_count": AggregateStats(count=1, mean=4.0, std=0.0, min=4.0, max=4.0),
                    "llm_prompt_tokens": AggregateStats(count=1, mean=100.0, std=0.0, min=100.0, max=100.0),
                    "llm_completion_tokens": AggregateStats(count=1, mean=50.0, std=0.0, min=50.0, max=50.0),
                    "llm_total_tokens": AggregateStats(count=1, mean=150.0, std=0.0, min=150.0, max=150.0),
                    "llm_usage_missing_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "llm_latency_ms": AggregateStats(count=1, mean=50.0, std=0.0, min=50.0, max=50.0),
                    "latency_ms": AggregateStats(count=1, mean=200.0, std=0.0, min=200.0, max=200.0),
                },
            ),
            "B": ExperimentSummary(
                experiment=ExperimentName.B,
                run_count=1,
                success_count=1,
                failure_count=0,
                success_rate=1.0,
                metrics={
                    "message_count": AggregateStats(count=1, mean=8.0, std=0.0, min=8.0, max=8.0),
                    "text_chars": AggregateStats(count=1, mean=4000.0, std=0.0, min=4000.0, max=4000.0),
                    "text_tokens": AggregateStats(count=1, mean=800.0, std=0.0, min=800.0, max=800.0),
                    "protocol_bytes": AggregateStats(count=1, mean=1500.0, std=0.0, min=1500.0, max=1500.0),
                    "state_transfer_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "state_bytes": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "artifact_ref_count": AggregateStats(count=1, mean=1.0, std=0.0, min=1.0, max=1.0),
                    "memory_retrieved": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_used": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_effective": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_harmful": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "tool_call_count": AggregateStats(count=1, mean=3.0, std=0.0, min=3.0, max=3.0),
                    "repeated_tool_call_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "llm_call_count": AggregateStats(count=1, mean=4.0, std=0.0, min=4.0, max=4.0),
                    "llm_prompt_tokens": AggregateStats(count=1, mean=100.0, std=0.0, min=100.0, max=100.0),
                    "llm_completion_tokens": AggregateStats(count=1, mean=50.0, std=0.0, min=50.0, max=50.0),
                    "llm_total_tokens": AggregateStats(count=1, mean=150.0, std=0.0, min=150.0, max=150.0),
                    "llm_usage_missing_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "llm_latency_ms": AggregateStats(count=1, mean=50.0, std=0.0, min=50.0, max=50.0),
                    "latency_ms": AggregateStats(count=1, mean=180.0, std=0.0, min=180.0, max=180.0),
                },
            ),
            "C": ExperimentSummary(
                experiment=ExperimentName.C,
                run_count=1,
                success_count=1,
                failure_count=0,
                success_rate=1.0,
                metrics={
                    "message_count": AggregateStats(count=1, mean=8.0, std=0.0, min=8.0, max=8.0),
                    "text_chars": AggregateStats(count=1, mean=3500.0, std=0.0, min=3500.0, max=3500.0),
                    "text_tokens": AggregateStats(count=1, mean=700.0, std=0.0, min=700.0, max=700.0),
                    "protocol_bytes": AggregateStats(count=1, mean=1400.0, std=0.0, min=1400.0, max=1400.0),
                    "state_transfer_count": AggregateStats(count=1, mean=2.0, std=0.0, min=2.0, max=2.0),
                    "state_bytes": AggregateStats(count=1, mean=1024.0, std=0.0, min=1024.0, max=1024.0),
                    "artifact_ref_count": AggregateStats(count=1, mean=1.0, std=0.0, min=1.0, max=1.0),
                    "memory_retrieved": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_used": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_effective": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "memory_harmful": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "tool_call_count": AggregateStats(count=1, mean=3.0, std=0.0, min=3.0, max=3.0),
                    "repeated_tool_call_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "llm_call_count": AggregateStats(count=1, mean=4.0, std=0.0, min=4.0, max=4.0),
                    "llm_prompt_tokens": AggregateStats(count=1, mean=100.0, std=0.0, min=100.0, max=100.0),
                    "llm_completion_tokens": AggregateStats(count=1, mean=50.0, std=0.0, min=50.0, max=50.0),
                    "llm_total_tokens": AggregateStats(count=1, mean=150.0, std=0.0, min=150.0, max=150.0),
                    "llm_usage_missing_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "llm_latency_ms": AggregateStats(count=1, mean=50.0, std=0.0, min=50.0, max=50.0),
                    "latency_ms": AggregateStats(count=1, mean=170.0, std=0.0, min=170.0, max=170.0),
                },
            ),
            "D": ExperimentSummary(
                experiment=ExperimentName.D,
                run_count=1,
                success_count=1,
                failure_count=0,
                success_rate=1.0,
                metrics={
                    "message_count": AggregateStats(count=1, mean=8.0, std=0.0, min=8.0, max=8.0),
                    "text_chars": AggregateStats(count=1, mean=3000.0, std=0.0, min=3000.0, max=3000.0),
                    "text_tokens": AggregateStats(count=1, mean=600.0, std=0.0, min=600.0, max=600.0),
                    "protocol_bytes": AggregateStats(count=1, mean=1300.0, std=0.0, min=1300.0, max=1300.0),
                    "state_transfer_count": AggregateStats(count=1, mean=2.0, std=0.0, min=2.0, max=2.0),
                    "state_bytes": AggregateStats(count=1, mean=1024.0, std=0.0, min=1024.0, max=1024.0),
                    "artifact_ref_count": AggregateStats(count=1, mean=1.0, std=0.0, min=1.0, max=1.0),
                    "memory_retrieved": AggregateStats(count=1, mean=5.0, std=0.0, min=5.0, max=5.0),
                    "memory_used": AggregateStats(count=1, mean=10.0, std=0.0, min=10.0, max=10.0),
                    "memory_effective": AggregateStats(count=1, mean=8.0, std=0.0, min=8.0, max=8.0),
                    "memory_harmful": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "tool_call_count": AggregateStats(count=1, mean=3.0, std=0.0, min=3.0, max=3.0),
                    "repeated_tool_call_count": AggregateStats(count=1, mean=1.0, std=0.0, min=1.0, max=1.0),
                    "llm_call_count": AggregateStats(count=1, mean=4.0, std=0.0, min=4.0, max=4.0),
                    "llm_prompt_tokens": AggregateStats(count=1, mean=100.0, std=0.0, min=100.0, max=100.0),
                    "llm_completion_tokens": AggregateStats(count=1, mean=50.0, std=0.0, min=50.0, max=50.0),
                    "llm_total_tokens": AggregateStats(count=1, mean=150.0, std=0.0, min=150.0, max=150.0),
                    "llm_usage_missing_count": AggregateStats(count=1, mean=0.0, std=0.0, min=0.0, max=0.0),
                    "llm_latency_ms": AggregateStats(count=1, mean=50.0, std=0.0, min=50.0, max=50.0),
                    "latency_ms": AggregateStats(count=1, mean=150.0, std=0.0, min=150.0, max=150.0),
                },
            ),
        },
        derived={
            "B_vs_A": DerivedMetrics(
                token_saving_rate=0.20,
                char_saving_rate=0.20,
                latency_improvement_rate=0.10,
                repeat_work_reduction_rate=None,
                effective_hit_rate=None,
            ),
            "C_vs_B": DerivedMetrics(
                token_saving_rate=0.125,
                char_saving_rate=0.125,
                latency_improvement_rate=0.055555555,
                repeat_work_reduction_rate=None,
                effective_hit_rate=None,
            ),
            "D_vs_C": DerivedMetrics(
                token_saving_rate=0.142857142,
                char_saving_rate=0.142857142,
                latency_improvement_rate=0.117647058,
                repeat_work_reduction_rate=None,
                effective_hit_rate=0.8,
            ),
        },
    )


class TestRenderMarkdownReport:
    """Test render_markdown_report function."""

    def test_returns_str(self):
        """render_markdown_report returns str."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        assert isinstance(report, str)

    def test_final_newline(self):
        """Report ends with newline."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        assert report.endswith("\n")

    def test_title_present(self):
        """Report contains title."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        assert "# AgentIPC Benchmark Report" in report

    def test_required_sections_present(self):
        """All required sections are present."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        required_sections = [
            "## Overview",
            "## Communication",
            "## State",
            "## Memory",
            "## LLM Usage",
            "## Latency",
            "## Derived Comparisons",
            "## Interpretation Notes",
        ]

        for section in required_sections:
            assert section in report, f"Missing section: {section}"

    def test_communication_table_has_abcd_rows(self):
        """Communication table has A, B, C, D rows."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        # Find Communication section
        comm_start = report.index("## Communication")
        comm_section = report[comm_start:comm_start + 1000]

        assert "| A |" in comm_section
        assert "| B |" in comm_section
        assert "| C |" in comm_section
        assert "| D |" in comm_section

    def test_state_table_has_abcd_rows(self):
        """State table has A, B, C, D rows."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        state_start = report.index("## State")
        state_section = report[state_start:state_start + 1000]

        assert "| A |" in state_section
        assert "| B |" in state_section
        assert "| C |" in state_section
        assert "| D |" in state_section

    def test_memory_table_has_abcd_rows(self):
        """Memory table has A, B, C, D rows."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        memory_start = report.index("## Memory")
        memory_section = report[memory_start:memory_start + 1000]

        assert "| A |" in memory_section
        assert "| B |" in memory_section
        assert "| C |" in memory_section
        assert "| D |" in memory_section

    def test_llm_usage_table_has_abcd_rows(self):
        """LLM Usage table has A, B, C, D rows."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        llm_start = report.index("## LLM Usage")
        llm_section = report[llm_start:llm_start + 1000]

        assert "| A |" in llm_section
        assert "| B |" in llm_section
        assert "| C |" in llm_section
        assert "| D |" in llm_section

    def test_latency_table_has_abcd_rows(self):
        """Latency table has A, B, C, D rows."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        latency_start = report.index("## Latency")
        latency_section = report[latency_start:latency_start + 1000]

        assert "| A |" in latency_section
        assert "| B |" in latency_section
        assert "| C |" in latency_section
        assert "| D |" in latency_section

    def test_derived_comparison_three_rows(self):
        """Derived Comparisons has exactly 3 rows."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        derived_start = report.index("## Derived Comparisons")
        derived_end = report.index("## Interpretation Notes")
        derived_section = report[derived_start:derived_end]

        assert "| B vs A |" in derived_section
        assert "| C vs B |" in derived_section
        assert "| D vs C |" in derived_section

    def test_none_rendered_as_na(self):
        """None values are rendered as N/A."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        # B_vs_A has effective_hit_rate=None
        derived_start = report.index("## Derived Comparisons")
        derived_end = report.index("## Interpretation Notes")
        derived_section = report[derived_start:derived_end]

        # B vs A row should have N/A for effective hit rate
        assert "N/A" in derived_section

    def test_positive_rate_formatted_as_percentage(self):
        """Positive rates are formatted as xx.xx%."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        # B_vs_A token_saving_rate=0.20 should be 20.00%
        assert "20.00%" in report

    def test_negative_rate_preserves_sign(self):
        """Negative rates preserve negative sign."""
        summary = _build_minimal_summary()
        # Modify to have a negative rate
        summary.derived["B_vs_A"] = DerivedMetrics(
            token_saving_rate=-0.10,  # 10% regression
            char_saving_rate=0.20,
            latency_improvement_rate=0.10,
            repeat_work_reduction_rate=None,
            effective_hit_rate=None,
        )

        report = render_markdown_report(summary)

        assert "-10.00%" in report

    def test_interpretation_notes_distinguish_text_tokens_from_llm_tokens(self):
        """Interpretation notes clarify text_tokens vs llm_total_tokens."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        notes_start = report.index("## Interpretation Notes")
        notes_section = report[notes_start:]

        assert "text_tokens" in notes_section
        assert "llm_total_tokens" in notes_section
        assert "TextCounter" in notes_section or "communication-side" in notes_section

    def test_interpretation_notes_explain_protocol_bytes_vs_text_chars(self):
        """Interpretation notes explain protocol_bytes vs text_chars."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        notes_start = report.index("## Interpretation Notes")
        notes_section = report[notes_start:]

        assert "protocol_bytes" in notes_section
        assert "text_chars" in notes_section

    def test_interpretation_notes_explain_derived_ablations(self):
        """Interpretation notes explain B vs A, C vs B, D vs C."""
        summary = _build_minimal_summary()
        report = render_markdown_report(summary)

        notes_start = report.index("## Interpretation Notes")
        notes_section = report[notes_start:]

        assert "B vs A" in notes_section
        assert "C vs B" in notes_section
        assert "D vs C" in notes_section

    def test_wrong_summary_type_rejected(self):
        """Non-BenchmarkSummary is rejected."""
        with pytest.raises(TypeError, match="summary must be a BenchmarkSummary"):
            render_markdown_report({"not": "a summary"})  # type: ignore[arg-type]


class TestWriteMarkdownReport:
    """Test write_markdown_report function."""

    def test_report_written_as_utf8(self, tmp_path: Path):
        """Report is written as UTF-8."""
        report_path = tmp_path / "report.md"
        summary = _build_minimal_summary()

        write_markdown_report(report_path, summary)

        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "# AgentIPC Benchmark Report" in content

    def test_write_refuses_overwrite(self, tmp_path: Path):
        """write_markdown_report refuses to overwrite existing file."""
        report_path = tmp_path / "report.md"
        summary = _build_minimal_summary()

        # First write succeeds
        write_markdown_report(report_path, summary)

        # Second write fails
        with pytest.raises(FileExistsError):
            write_markdown_report(report_path, summary)

    def test_wrong_path_type_rejected(self, tmp_path: Path):
        """Non-str/Path path is rejected."""
        summary = _build_minimal_summary()

        with pytest.raises(TypeError, match="path must be a str or Path"):
            write_markdown_report(123, summary)  # type: ignore[arg-type]

    def test_wrong_summary_type_rejected(self, tmp_path: Path):
        """Non-BenchmarkSummary summary is rejected."""
        with pytest.raises(TypeError, match="summary must be a BenchmarkSummary"):
            write_markdown_report(tmp_path / "report.md", "not a summary")  # type: ignore[arg-type]
