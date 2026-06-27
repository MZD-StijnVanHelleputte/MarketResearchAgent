from reports.assembler import ReportDraft
from reports.citations import build_citation_index, render_sources_section, replace_citation_markers


def _draft(**overrides) -> ReportDraft:
    return ReportDraft(
        run_id="test-run",
        query="q",
        exec_summary=overrides.get("exec_summary", ""),
        chapters=overrides.get("chapters", []),
        merge_log=[],
        warnings=[],
        citation_registry=overrides.get("citation_registry", {}),
    )


_REGISTRY = {
    1: {"id": 1, "title": "CAT Q3 results", "url": "https://a.com", "publisher": "Reuters"},
    2: {"id": 2, "title": "Copper outlook", "url": "https://b.com", "publisher": "MINING.COM"},
    3: {"id": 3, "title": "Mining Metals (Alpha Vantage)", "url": None, "publisher": None},
}


def test_build_citation_index_numbers_in_first_occurrence_order():
    draft = _draft(
        chapters=[
            {"domain": "competition", "text": "CAT grew [Source: 1]."},
            {"domain": "commodities", "text": "Copper rose [Source: 2]."},
        ],
        citation_registry=_REGISTRY,
    )
    index = build_citation_index(draft, _REGISTRY)
    assert index == {1: 1, 2: 2}


def test_build_citation_index_dedupes_repeated_citation():
    draft = _draft(
        chapters=[
            {"domain": "competition", "text": "CAT grew [Source: 1]. Margins held [Source: 1]."},
        ],
        citation_registry=_REGISTRY,
    )
    index = build_citation_index(draft, _REGISTRY)
    assert index == {1: 1}


def test_build_citation_index_scans_subchapters_and_exec_summary():
    draft = _draft(
        exec_summary="Overview [Source: 3].",
        chapters=[
            {"domain": "competition", "text": "Rollup.", "subchapters": [
                {"text": "Entity detail [Source: 1]."},
            ]},
        ],
        citation_registry=_REGISTRY,
    )
    index = build_citation_index(draft, _REGISTRY)
    assert index == {3: 1, 1: 2}


def test_build_citation_index_skips_unknown_id():
    draft = _draft(chapters=[{"domain": "competition", "text": "CAT grew [Source: 999]."}])
    index = build_citation_index(draft, {})
    assert index == {}


def test_replace_citation_markers_inserts_superscript():
    index = {1: 3}
    out = replace_citation_markers("CAT grew [Source: 1].", index)
    assert out == "CAT grew <super>[3]</super>."


def test_replace_citation_markers_drops_unknown_citation():
    out = replace_citation_markers("CAT grew [Source: 999].", {})
    assert out == "CAT grew ."


def test_render_sources_section_empty_index_returns_nothing():
    from reports.pdf_generator import _build_styles
    assert render_sources_section({}, {}, _build_styles()) == []


def test_render_sources_section_renders_title_and_publisher_with_link():
    from reportlab.platypus import Paragraph

    from reports.pdf_generator import _build_styles
    index = {1: 1, 3: 2}
    flowables = render_sources_section(index, _REGISTRY, _build_styles())
    paragraphs = [f for f in flowables if isinstance(f, Paragraph)]
    texts = [p.text for p in paragraphs]
    assert any("<link" in t and "https://a.com" in t and "CAT Q3 results" in t and "Reuters" in t for t in texts)
    # No url/publisher → bare title, no link wrapper, never a raw tool/function name
    assert any(t.startswith("[2] Mining Metals (Alpha Vantage)") and "<link" not in t for t in texts)
