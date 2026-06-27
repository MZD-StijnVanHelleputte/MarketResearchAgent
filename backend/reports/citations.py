"""Converts inline "[Source: <id>]" markers into numbered academic-style footnotes.

The synthesis prompts ask the LLM to ground claims with inline `[Source: <id>]`
markers, where <id> is a global citation id from the run's citation registry
(see prompts/synthesize_prompt.py, prompts/domain_agent_prompt.py, and
core/merger.py::assign_global_citation_ids — the LLM never invents a citation's
identity, only references one by number). Rather than rendering those markers
as literal inline text, the PDF generator numbers each unique citation on first
occurrence and replaces the marker with a superscript reference; the full
bibliography entries (title, publisher, clickable url) are listed once in a
"Sources" section at the end of the report.
"""
import re

from reports.assembler import ReportDraft

_MARKER_RE = re.compile(r"\[Source:\s*(\d+)\s*\]")


def build_citation_index(draft: ReportDraft, registry: dict[int, dict]) -> dict[int, int]:
    """Map each global citation id actually referenced to a display number,
    assigned sequentially in first-occurrence (reading) order. Ids referenced in
    the text but absent from the registry are skipped."""
    index: dict[int, int] = {}

    def _scan(text: str) -> None:
        for match in _MARKER_RE.finditer(text or ""):
            cid = int(match.group(1))
            if cid in registry and cid not in index:
                index[cid] = len(index) + 1

    _scan(draft.exec_summary)
    for chapter in draft.chapters:
        _scan(chapter.get("text") or "")
        for sub in chapter.get("subchapters") or []:
            _scan(sub.get("text") or "")

    return index


def replace_citation_markers(text: str, index: dict[int, int]) -> str:
    """Replace "[Source: id]" markers with superscript footnote numbers, e.g. <super>[3]</super>."""

    def _sub(match: re.Match) -> str:
        cid = int(match.group(1))
        number = index.get(cid)
        return f"<super>[{number}]</super>" if number else ""

    return _MARKER_RE.sub(_sub, text or "")


def render_sources_section(index: dict[int, int], registry: dict[int, dict], styles: dict) -> list:
    """Build a "Sources" heading + one numbered bibliography entry per cited source."""
    from reportlab.lib import colors
    from reportlab.platypus import HRFlowable, Paragraph

    from reports.markup import escape_text

    if not index:
        return []

    flowables: list = [
        Paragraph("Sources", styles["appendix_heading"]),
        HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=6),
    ]
    for cid, number in sorted(index.items(), key=lambda kv: kv[1]):
        record = registry.get(cid, {})
        title = record.get("title") or "Unknown source"
        publisher = record.get("publisher")
        url = record.get("url")
        label = f"{title} — {publisher}" if publisher else title
        safe = escape_text(label)
        if url:
            body = f'<link href="{escape_text(url)}">{safe}</link>'
        else:
            body = safe
        flowables.append(Paragraph(f"[{number}] {body}", styles["appendix_item"]))
    return flowables
