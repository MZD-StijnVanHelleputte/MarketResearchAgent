"""Converts LLM/code-authored markdown into ReportLab's mini-XML Paragraph markup.

LLM-authored chapter text (and the code-inserted "**Key figures**" block in
agents/synthesis_agent.py) uses markdown emphasis, but reportlab's `Paragraph` only
understands its own tags (`<b>`, `<i>`, `<super>`, ...). Passed through unescaped, raw
`**bold**` shows up as literal asterisks, and any `&`/`<`/`>` in the source text can
break Paragraph's XML parser outright.
"""
import html
import re

from reports.citations import replace_citation_markers

_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\w)[*_](?!\s)(.+?)(?<!\s)[*_](?!\w)")


def escape_text(text: str) -> str:
    """Escape &, <, > so arbitrary text is safe to embed in a ReportLab Paragraph."""
    return html.escape(text or "", quote=False)


def render_text(text: str, citation_index: dict[str, int]) -> str:
    """Escape, resolve citation footnotes, then convert markdown emphasis to ReportLab markup.

    Order matters: escape raw text first (so user-authored "<"/"&" can't collide with
    tags inserted below), then insert footnote/bold/italic tags last.
    """
    escaped = escape_text(text)
    with_citations = replace_citation_markers(escaped, citation_index)
    bolded = _BOLD_RE.sub(r"<b>\1</b>", with_citations)
    return _ITALIC_RE.sub(r"<i>\1</i>", bolded)
