from reportlab.platypus import Paragraph

from reports.markup import escape_text, render_text


def test_escape_text_escapes_special_chars():
    assert escape_text("Caterpillar & Co <2026>") == "Caterpillar &amp; Co &lt;2026&gt;"


def test_render_text_converts_bold():
    assert render_text("This is **important**.", {}) == "This is <b>important</b>."


def test_render_text_converts_italic():
    assert render_text("This is _subtle_.", {}) == "This is <i>subtle</i>."


def test_render_text_escapes_before_inserting_tags():
    out = render_text("Komatsu & Caterpillar compared **directly**.", {})
    assert out == "Komatsu &amp; Caterpillar compared <b>directly</b>."
    # Output must remain valid mini-XML for ReportLab.
    Paragraph(out, getSampleStyle())


def test_render_text_resolves_citation_then_bold():
    index = {1: 1}
    out = render_text("CAT grew **fast** [Source: 1].", index)
    assert out == "CAT grew <b>fast</b> <super>[1]</super>."
    Paragraph(out, getSampleStyle())


def test_render_text_survives_unescaped_ampersand_in_paragraph():
    """Regression: unescaped & previously crashed ReportLab's Paragraph XML parser."""
    out = render_text("Komatsu & Caterpillar", {})
    Paragraph(out, getSampleStyle())  # must not raise


def getSampleStyle():
    from reportlab.lib.styles import getSampleStyleSheet
    return getSampleStyleSheet()["Normal"]
