"""Tests for self-healing FRED series_id resolution: when a planner-guessed
series_id is rejected by FRED with a permanent 400 ("series does not exist"),
the domain agent should recover via FRED's own discovery endpoints
(series/search, falling back to tags/series) before giving up. Recovery only
accepts a popular, recently-updated series; otherwise it returns None so the
agent falls back to web search instead of reporting an off-topic series."""
import datetime as dt
from unittest.mock import AsyncMock, patch

import pytest

from agents import make_domain_agent

_RECENT = (dt.date.today() - dt.timedelta(days=30)).isoformat()
_STALE = (dt.date.today() - dt.timedelta(days=3000)).isoformat()


def _candidate(series_id: str, popularity: int = 80, observation_end: str = _RECENT) -> dict:
    return {
        "series_id": series_id,
        "title": series_id,
        "popularity": popularity,
        "observation_end": observation_end,
    }


def _permanent_series_error(series_id: str) -> Exception:
    from clients.base_http_client import ClientError

    try:
        try:
            raise ClientError(400, "Bad Request. The series does not exist.")
        except ClientError as inner:
            raise Exception(f"FRED observations failed for '{series_id}'") from inner
    except Exception as exc:
        return exc


@pytest.mark.asyncio
async def test_resolves_via_series_search_and_retries():
    """search_fred_series finds a real id; the original tool is retried with it."""
    good = {"series_id": "INDPRO", "title": "Industrial Production Index", "observations": []}

    async def route(tool_name, args, call_id=None, count_failures=True):
        if tool_name == "get_macro_indicator" and args.get("series_id") == "STEEL_PRODUCTION_INDEX":
            raise _permanent_series_error("STEEL_PRODUCTION_INDEX")
        if tool_name == "search_fred_series":
            return {"results": [_candidate("INDPRO")], "count": 1}
        if tool_name == "get_macro_indicator" and args.get("series_id") == "INDPRO":
            return good
        raise AssertionError(f"unexpected call: {tool_name}({args})")

    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=AsyncMock(side_effect=route)):
        agent = make_domain_agent("macroeconomics")
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicator", {"series_id": "STEEL_PRODUCTION_INDEX"}, "c1"
        )

    assert err is None
    assert result == good


@pytest.mark.asyncio
async def test_falls_back_to_tags_series_when_search_is_empty():
    """search_fred_series returns nothing; get_fred_series_by_tags recovers an id."""
    good = {"series_id": "WPU101", "title": "PPI: Steel", "observations": []}

    async def route(tool_name, args, call_id=None, count_failures=True):
        if tool_name == "get_macro_indicator" and args.get("series_id") == "STEEL_PRICE_INDEX":
            raise _permanent_series_error("STEEL_PRICE_INDEX")
        if tool_name == "search_fred_series":
            return {"results": [], "count": 0}
        if tool_name == "get_fred_series_by_tags":
            assert args["tag_names"] == "steel;price;index"
            return {"results": [_candidate("WPU101", popularity=60)], "count": 1}
        if tool_name == "get_macro_indicator" and args.get("series_id") == "WPU101":
            return good
        raise AssertionError(f"unexpected call: {tool_name}({args})")

    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=AsyncMock(side_effect=route)):
        agent = make_domain_agent("macroeconomics")
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicator", {"series_id": "STEEL_PRICE_INDEX"}, "c1"
        )

    assert err is None
    assert result == good


@pytest.mark.asyncio
async def test_gives_up_when_both_discovery_endpoints_are_empty():
    """Neither discovery endpoint finds a match — original failure is preserved."""

    async def route(tool_name, args, call_id=None, count_failures=True):
        if tool_name == "get_macro_indicator":
            raise _permanent_series_error("NONSENSE_ID")
        if tool_name in ("search_fred_series", "get_fred_series_by_tags"):
            return {"results": [], "count": 0}
        raise AssertionError(f"unexpected call: {tool_name}({args})")

    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=AsyncMock(side_effect=route)):
        agent = make_domain_agent("macroeconomics")
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicator", {"series_id": "NONSENSE_ID"}, "c1"
        )

    assert result is None
    assert err is not None
    assert "NONSENSE_ID" in err


@pytest.mark.asyncio
async def test_low_popularity_or_irrelevant_match_is_rejected():
    """A vague search whose only hits are obscure/low-popularity series (e.g. 'tariff'
    → 'Coffee Imports') must NOT be accepted; recovery returns None and the original
    tool is never retried (the agent should fall back to web search)."""
    retried = []

    async def route(tool_name, args, call_id=None, count_failures=True):
        if tool_name == "get_macro_indicator" and args.get("series_id") == "US_TARIFF_RATE":
            raise _permanent_series_error("US_TARIFF_RATE")
        if tool_name == "search_fred_series":
            return {"results": [_candidate("M07038USM149NNBR", popularity=6)], "count": 1}
        if tool_name == "get_fred_series_by_tags":
            return {"results": [_candidate("BOGZ1FA103090603A", popularity=0)], "count": 1}
        if tool_name == "get_macro_indicator":
            retried.append(args.get("series_id"))
            return {"series_id": args.get("series_id"), "observations": []}
        raise AssertionError(f"unexpected call: {tool_name}({args})")

    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=AsyncMock(side_effect=route)):
        agent = make_domain_agent("macroeconomics")
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicator", {"series_id": "US_TARIFF_RATE"}, "c1"
        )

    assert result is None
    assert err is not None
    assert retried == []  # never retried with a garbage series


@pytest.mark.asyncio
async def test_picks_most_popular_recent_candidate():
    """Among viable hits, the most popular is chosen; discontinued (stale) series are
    skipped even if highly popular."""
    async def route(tool_name, args, call_id=None, count_failures=True):
        if tool_name == "get_macro_indicator" and args.get("series_id") == "INFRA_SPEND":
            raise _permanent_series_error("INFRA_SPEND")
        if tool_name == "search_fred_series":
            return {"results": [
                _candidate("STALE_BIG", popularity=99, observation_end=_STALE),
                _candidate("TTLCONS", popularity=70),
                _candidate("OTHER", popularity=40),
            ], "count": 3}
        if tool_name == "get_macro_indicator" and args.get("series_id") == "TTLCONS":
            return {"series_id": "TTLCONS", "observations": []}
        raise AssertionError(f"unexpected call: {tool_name}({args})")

    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=AsyncMock(side_effect=route)):
        agent = make_domain_agent("macroeconomics")
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicator", {"series_id": "INFRA_SPEND"}, "c1"
        )

    assert err is None
    assert result["series_id"] == "TTLCONS"


@pytest.mark.asyncio
async def test_non_fred_tool_never_triggers_discovery():
    """A permanent error on an unrelated tool must not call FRED discovery endpoints."""
    from clients.base_http_client import ClientError

    def _filing_error() -> Exception:
        try:
            try:
                raise ClientError(404, "No filings found")
            except ClientError as inner:
                raise Exception("SEC EDGAR lookup failed") from inner
        except Exception as exc:
            return exc

    route = AsyncMock(side_effect=_filing_error())
    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=route):
        agent = make_domain_agent("macroeconomics")
        result, usages, err = await agent._call_tool_with_repair(
            "search_sec_filings", {"ticker": "CAT"}, "c1"
        )

    assert result is None
    assert err is not None
    assert route.call_count == 1
