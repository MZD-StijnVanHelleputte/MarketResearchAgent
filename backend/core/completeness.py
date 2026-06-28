"""Pre-finalization completeness check for synthesized chapters/subchapters.

`synthesize_node` (core/graph.py) assembles whatever the per-domain/per-entity
synthesis produced, with no check that it actually produced something. When a
Tier-1 subchapter falls back (CrewAI failure, or genuinely no evidence), the
report renders the literal placeholder ("Limited evidence collected for X")
verbatim. This module gives the synthesize_node a way to notice that before
the report is built, and to write an honest message instead of a generic one
once a remediation attempt has been exhausted.
"""
from __future__ import annotations

FALLBACK_MARKERS = (
    "Limited evidence collected for",
    "No data collected for",
    "No tool calls assigned to",
)


def has_evidence(item: dict) -> bool:
    """True if a subchapter/chapter dict carries any figures, datasets, or citations."""
    return bool(item.get("figures") or item.get("datasets") or item.get("citations"))


def is_fallback_text(text: str) -> bool:
    return any(marker in text for marker in FALLBACK_MARKERS)


def is_gap(item: dict) -> bool:
    """A chapter/subchapter is a completeness gap if synthesis fell back to the
    generic placeholder text.

    Deliberately does NOT treat "no figures/datasets/citations" alone as a gap:
    thematic domains (general_search) legitimately synthesize
    real prose from retrieved chunks with no structured evidence attached, and
    flagging those would trigger needless remediation churn. `has_evidence` is
    still useful as a *strategy* signal once something is already a known gap
    (whether to also try a broadened search before resynthesizing).
    """
    return is_fallback_text(item.get("text", ""))


def broadened_query(label: str, base_query: str) -> str:
    """A looser web-search query than the entity's own query_hint, used for one
    extra remediation attempt when the first collection pass found nothing."""
    return f"{label} {base_query} latest news analysis"


def honest_fallback_message(label: str, tool_errors: list[str] | None = None) -> str:
    """Replace the generic 'Limited evidence collected for X' with a specific,
    diagnosable statement once remediation has been attempted and still failed."""
    if tool_errors:
        attempted = "; ".join(tool_errors[:3])
        return (
            f"No usable data could be retrieved for {label} in this run. "
            f"Attempted: {attempted}. See merge log for details."
        )
    return (
        f"No financial data, news, or analyst commentary could be retrieved for "
        f"{label} in this run, even after a retry. See merge log for details."
    )
