"""Unit tests for SynthesisAgent._append_figures (Round 2 dedup of figures vs prose)."""
from agents.synthesis_agent import SynthesisAgent


def test_appends_figure_not_mentioned_in_text():
    text = "Caterpillar had a strong quarter."
    out = SynthesisAgent._append_figures(text, {"CAT_revenue_bn": "67.1"})
    assert "**Key figures**" in out
    assert "CAT_revenue_bn: 67.1" in out


def test_does_not_append_figure_already_in_text():
    text = "Caterpillar posted revenue of 67.1 billion dollars."
    out = SynthesisAgent._append_figures(text, {"CAT_revenue_bn": "67.1"})
    assert out == text


def test_appends_only_missing_figures_when_mixed():
    text = "Caterpillar posted revenue of 67.1 billion; margin details were not discussed."
    out = SynthesisAgent._append_figures(
        text, {"CAT_revenue_bn": "67.1", "CAT_margin": "19%"}
    )
    assert "CAT_margin: 19%" in out
    assert "CAT_revenue_bn: 67.1" not in out.split("**Key figures**")[1]


def test_no_figures_returns_text_unchanged():
    text = "No numbers here."
    assert SynthesisAgent._append_figures(text, {}) == text


def test_existing_key_figures_block_short_circuits():
    text = "Some prose.\n\n**Key figures**\n- already: here"
    out = SynthesisAgent._append_figures(text, {"new_metric": "42"})
    assert out == text
