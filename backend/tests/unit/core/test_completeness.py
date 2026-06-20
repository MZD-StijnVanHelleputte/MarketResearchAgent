"""Unit tests for core/completeness.py (pre-finalization gap detection)."""
from core import completeness


def test_fallback_text_is_a_gap():
    sc = {"text": "Limited evidence collected for RIO.", "figures": {"x": "1"}, "datasets": [], "citations": []}
    assert completeness.is_gap(sc)


def test_real_prose_with_no_structured_evidence_is_not_a_gap():
    # Thematic domains (general_search, macro_geopolitics) legitimately synthesize
    # from retrieved chunks alone, with no figures/datasets/citations attached.
    sc = {"text": "EV adoption is accelerating copper demand globally.", "figures": {}, "datasets": [], "citations": []}
    assert not completeness.is_gap(sc)


def test_healthy_subchapter_is_not_a_gap():
    sc = {
        "text": "RIO had a strong quarter, driven by iron ore prices.",
        "figures": {"rio_revenue_bn": "54.6"},
        "datasets": [{"tool": "equity_history", "rows": []}],
        "citations": ["https://example.com/rio"],
    }
    assert not completeness.is_gap(sc)


def test_has_evidence_checks_any_of_three_fields():
    assert completeness.has_evidence({"figures": {}, "datasets": [], "citations": ["url"]})
    assert not completeness.has_evidence({"figures": {}, "datasets": [], "citations": []})


def test_honest_fallback_message_includes_tool_errors():
    msg = completeness.honest_fallback_message("Escondida", ["web_search: 429 rate limited"])
    assert "Escondida" in msg
    assert "web_search: 429 rate limited" in msg


def test_honest_fallback_message_without_tool_errors():
    msg = completeness.honest_fallback_message("Escondida", None)
    assert "Escondida" in msg
    assert "retry" in msg


def test_broadened_query_includes_label_and_base_query():
    q = completeness.broadened_query("Escondida", "copper demand 2026")
    assert "Escondida" in q
    assert "copper demand 2026" in q
