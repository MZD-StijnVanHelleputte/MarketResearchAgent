"""Helpers for deciding whether a tool response contains usable evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Number


_LIST_EVIDENCE_KEYS = {
    "articles",
    "results",
    "rows",
    "pages",
    "filings",
    "statements",
    "ratios",
    "estimates",
    "surprises",
    "ratings",
    "press_releases",
    "releases",
    "sources",
}

_SCALAR_EVIDENCE_KEYS = {
    "price",
    "price_usd",
    "market_cap",
    "market_cap_usd",
    "revenue",
    "revenue_usd",
    "net_income",
    "net_income_usd",
    "capex",
    "capex_usd",
    "pe_ratio",
    "value",
    "close",
    "open",
    "high",
    "low",
    "volume",
}


def has_usable_data(result: object) -> bool:
    """Return True when a tool result contains evidence useful for synthesis.

    Transport success is not enough for collection progress: empty search results,
    empty article lists, empty time series, and metadata-only payloads should not be
    counted as successful intelligence.
    """
    if not isinstance(result, Mapping) or not result:
        return False

    if _latest_has_value(result.get("latest")):
        return True

    technical_report = result.get("technical_report")
    if isinstance(technical_report, Mapping) and _mapping_has_content(technical_report):
        return True

    report = result.get("report")
    citations = result.get("citations")
    if isinstance(report, str) and report.strip():
        return True
    if isinstance(citations, Sequence) and not isinstance(citations, (str, bytes)):
        if any(_item_has_content(item) for item in citations):
            return True

    saw_evidence_key = False
    for key in _LIST_EVIDENCE_KEYS:
        if key not in result:
            continue
        saw_evidence_key = True
        value = result.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if any(_item_has_content(item) for item in value):
                return True

    if any(result.get(key) is not None for key in _SCALAR_EVIDENCE_KEYS):
        return True

    count = result.get("count")
    if isinstance(count, Number) and count > 0:
        return True

    if saw_evidence_key:
        return False

    return False


def no_data_reason(tool_name: str) -> str:
    return f"{tool_name} returned no usable data"


def _latest_has_value(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(value.get(key) is not None for key in _SCALAR_EVIDENCE_KEYS)
    return value is not None


def _item_has_content(item: object) -> bool:
    if isinstance(item, Mapping):
        return _mapping_has_content(item)
    if isinstance(item, str):
        return bool(item.strip())
    return item is not None


def _mapping_has_content(value: Mapping) -> bool:
    for v in value.values():
        if isinstance(v, str) and v.strip():
            return True
        if isinstance(v, Number):
            return True
        if isinstance(v, Mapping) and _mapping_has_content(v):
            return True
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
            if any(_item_has_content(item) for item in v):
                return True
        if v is not None and not isinstance(v, (str, bytes, Mapping, Sequence)):
            return True
    return False
