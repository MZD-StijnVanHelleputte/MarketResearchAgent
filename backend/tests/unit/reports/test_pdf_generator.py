import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from reports.assembler import ReportDraft
from reports.pdf_generator import PdfGenerator, _dataset_flowables

_DEFAULT_MAX_ROWS = 10


def _minimal_draft(run_id: str = "test-run-pdf-001", **overrides) -> ReportDraft:
    return ReportDraft(
        run_id=run_id,
        query="What is Caterpillar's current capex cycle?",
        exec_summary=overrides.get("exec_summary", "This is a test executive summary."),
        chapters=overrides.get("chapters", [
            {"domain": "competition", "text": "CAT revenue grew 12% year-over-year."},
            {"domain": "commodities", "text": "Copper prices rose on supply disruptions."},
        ]),
        merge_log=overrides.get("merge_log", []),
        warnings=overrides.get("warnings", []),
    )


def test_generate_returns_pdf_bytes(tmp_path):
    draft = _minimal_draft()
    with patch("reports.pdf_generator.settings") as mock_settings:
        mock_settings.report.output_dir = str(tmp_path)
        result = PdfGenerator.generate(draft)
    assert isinstance(result, bytes)
    assert len(result) > 0
    assert result[:4] == b"%PDF"


def test_generate_writes_file_to_output_dir(tmp_path):
    draft = _minimal_draft(run_id="test-run-pdf-002")
    with patch("reports.pdf_generator.settings") as mock_settings:
        mock_settings.report.output_dir = str(tmp_path)
        PdfGenerator.generate(draft)
    expected = tmp_path / "test-run-pdf-002.pdf"
    assert expected.exists()
    assert expected.stat().st_size > 0


def test_generate_with_empty_exec_summary(tmp_path):
    draft = _minimal_draft(run_id="test-run-pdf-003", exec_summary="")
    with patch("reports.pdf_generator.settings") as mock_settings:
        mock_settings.report.output_dir = str(tmp_path)
        result = PdfGenerator.generate(draft)
    assert result[:4] == b"%PDF"


def test_generate_with_merge_log_includes_appendix(tmp_path):
    draft = _minimal_draft(
        run_id="test-run-pdf-004",
        merge_log=["Resolved contradiction in competition/capex: $14B vs $12B"],
        warnings=["Stale chunk detected in commodities (age: 95 days)"],
    )
    with patch("reports.pdf_generator.settings") as mock_settings:
        mock_settings.report.output_dir = str(tmp_path)
        result = PdfGenerator.generate(draft)
    assert result[:4] == b"%PDF"
    assert len(result) > 0


def test_generate_with_no_chapters(tmp_path):
    draft = _minimal_draft(run_id="test-run-pdf-005", chapters=[])
    with patch("reports.pdf_generator.settings") as mock_settings:
        mock_settings.report.output_dir = str(tmp_path)
        result = PdfGenerator.generate(draft)
    assert result[:4] == b"%PDF"


def test_generate_with_chapter_datasets_renders_table(tmp_path):
    draft = _minimal_draft(
        run_id="test-run-pdf-007",
        chapters=[
            {
                "domain": "competition",
                "text": "CAT revenue grew 12% year-over-year.",
                "datasets": [{
                    "tool": "get_equity_history",
                    "title": "CAT — 5y",
                    "kind": "table",
                    "columns": ["date", "close"],
                    "rows": [["2026-01-01", "350"], ["2026-01-02", "352"]],
                    "row_count": 2,
                }],
            },
        ],
    )
    with patch("reports.pdf_generator.settings") as mock_settings:
        mock_settings.report.output_dir = str(tmp_path)
        mock_settings.report.report_table_max_rows = _DEFAULT_MAX_ROWS
        result = PdfGenerator.generate(draft)
    assert result[:4] == b"%PDF"
    assert len(result) > 0


def test_dataset_flowables_skips_non_table_kind():
    assert _dataset_flowables({"kind": "list", "items": []}, {"table_caption": None}) == []


def test_dataset_flowables_skips_empty_rows():
    from reports.pdf_generator import _build_styles
    styles = _build_styles()
    ds = {"kind": "table", "title": "Empty", "columns": ["date"], "rows": []}
    assert _dataset_flowables(ds, styles) == []


def test_dataset_flowables_caps_rows_for_non_numeric_table():
    """A table with more rows than the threshold but no numeric column stays a (truncated) table."""
    from reports.pdf_generator import _build_styles
    styles = _build_styles()
    ds = {
        "kind": "table", "title": "Big table", "columns": ["id", "label"],
        "rows": [[f"id-{i}", f"label-{i}"] for i in range(_DEFAULT_MAX_ROWS + 10)],
    }
    flowables = _dataset_flowables(ds, styles)
    table = flowables[1]
    assert len(table._cellvalues) == _DEFAULT_MAX_ROWS + 1  # +1 header row


def test_dataset_flowables_charts_large_numeric_table():
    """A numeric table over the row threshold renders as an embedded chart image."""
    from reportlab.platypus import Image
    from reports.pdf_generator import _build_styles
    styles = _build_styles()
    ds = {
        "kind": "table", "title": "Copper price", "columns": ["date", "value (usd)"],
        "rows": [[f"2026-01-{i:02d}", str(100 + i)] for i in range(1, _DEFAULT_MAX_ROWS + 10)],
    }
    flowables = _dataset_flowables(ds, styles)
    assert isinstance(flowables[1], Image)


def test_dataset_flowables_falls_back_to_table_on_chart_render_failure():
    """If chart rendering fails, the dataset still renders as a (truncated) table."""
    from reports.pdf_generator import _build_styles
    styles = _build_styles()
    ds = {
        "kind": "table", "title": "Copper price", "columns": ["date", "value (usd)"],
        "rows": [[f"2026-01-{i:02d}", str(100 + i)] for i in range(1, _DEFAULT_MAX_ROWS + 10)],
    }
    with patch("reports.pdf_generator.render_chart", return_value=None):
        flowables = _dataset_flowables(ds, styles)
    table = flowables[1]
    assert len(table._cellvalues) == _DEFAULT_MAX_ROWS + 1


def test_build_dataset_owners_picks_higher_priority_domain():
    from reports.pdf_generator import _build_dataset_owners
    shared_ds = {
        "tool": "get_equity_history", "title": "BHP — 5y", "kind": "table",
        "columns": ["date", "close"], "rows": [["2026-01-01", "45"]],
        "series_id": "get_equity_history:BHP:5y",
    }
    chapters = [
        {"domain": "mining_projects", "datasets": [dict(shared_ds)]},
        {"domain": "customers", "datasets": [dict(shared_ds)]},
    ]
    owners = _build_dataset_owners(chapters)
    assert owners["get_equity_history:BHP:5y"] == "customers"


def test_build_dataset_owners_checks_subchapter_datasets_too():
    from reports.pdf_generator import _build_dataset_owners
    shared_ds = {
        "tool": "get_commodity_price", "title": "copper — 120 point(s)", "kind": "table",
        "columns": ["date", "value (usd)"], "rows": [["2026-01-01", "4.10"]],
        "series_id": "get_commodity_price:copper",
    }
    chapters = [
        {"domain": "mining_projects", "subchapters": [{"datasets": [dict(shared_ds)]}]},
        {"domain": "commodities", "datasets": [dict(shared_ds)]},
    ]
    owners = _build_dataset_owners(chapters)
    assert owners["get_commodity_price:copper"] == "commodities"


def test_generate_renders_shared_series_chart_only_once_across_domains(tmp_path):
    """A series collected by two domains should still produce a valid PDF, with only
    the higher-priority domain (per _dataset_flowables call count) rendering it."""
    shared_ds = {
        "tool": "get_equity_history", "title": "BHP — 5y", "kind": "table",
        "columns": ["date", "close"], "rows": [["2026-01-01", "45"], ["2026-01-02", "46"]],
        "series_id": "get_equity_history:BHP:5y",
    }
    draft = _minimal_draft(
        run_id="test-run-pdf-008",
        chapters=[
            {"domain": "mining_projects", "text": "Escondida update.", "datasets": [dict(shared_ds)]},
            {"domain": "customers", "text": "BHP capex update.", "datasets": [dict(shared_ds)]},
        ],
    )
    with patch("reports.pdf_generator.settings") as mock_settings, \
         patch("reports.pdf_generator._dataset_flowables", return_value=[]) as mock_flowables:
        mock_settings.report.output_dir = str(tmp_path)
        mock_settings.report.report_table_max_rows = _DEFAULT_MAX_ROWS
        result = PdfGenerator.generate(draft)
    assert result[:4] == b"%PDF"
    # customers outranks mining_projects in _DATASET_OWNERSHIP_PRIORITY, so only
    # its chapter should attempt to render the shared series.
    assert mock_flowables.call_count == 1


def test_generate_creates_output_dir_if_missing(tmp_path):
    nested = tmp_path / "deep" / "nested" / "dir"
    draft = _minimal_draft(run_id="test-run-pdf-006")
    with patch("reports.pdf_generator.settings") as mock_settings:
        mock_settings.report.output_dir = str(nested)
        PdfGenerator.generate(draft)
    assert (nested / "test-run-pdf-006.pdf").exists()
