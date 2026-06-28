"""Classify a raw tool result into a coarse (data_type, count) pair.

Single source of truth for "what kind of data did this tool return, and how
much of it" — used by the live tool_call event detail (core/tool_router) and the
provisional Sources rows (core/event_logger). The branch order and the resulting
data_type labels MUST stay aligned with BaseDomainAgent._to_datasets
(agents/base_domain_agent.py), which builds the full typed datasets for Gate 2;
this helper is the lightweight type+count view of the same shapes.
"""
from __future__ import annotations


def detect_data_type_and_count(result: object) -> tuple[str, int]:
    """Return ("numeric_series" | "financials" | "articles" | "web_results" |
    "filings" | "document" | "data", count). count is the number of points / rows
    / items the result carries (1 for single-document/opaque payloads, 0 if empty).
    """
    if not isinstance(result, dict):
        return "data", 0

    if "symbol" in result and ("latest" in result or "rows" in result):
        rows = [
            r for r in (result.get("rows") or [])
            if isinstance(r, dict) and r.get("value") is not None
        ]
        if not rows:
            latest = result.get("latest") or {}
            if isinstance(latest, dict) and latest.get("value") is not None:
                rows = [latest]
        return "numeric_series", len(rows)

    if "ticker" in result and "rows" in result:
        rows = [r for r in (result.get("rows") or []) if isinstance(r, dict)]
        return "financials", len(rows)

    if "articles" in result:
        return "articles", len(result.get("articles") or [])

    if "results" in result:
        return "web_results", len(result.get("results") or [])

    if "filings" in result:
        return "filings", len(result.get("filings") or [])

    if "technical_report" in result:
        return "document", 1

    # Opaque result: best-effort count of the first list value (e.g. a tool that
    # returns {"prices": [...]}), matching the legacy provisional-source behaviour.
    for val in result.values():
        if isinstance(val, list):
            return "data", len(val)
    return "data", 1 if result else 0
