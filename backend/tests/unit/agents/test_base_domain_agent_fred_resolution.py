"""Tests for self-healing FRED series_id resolution: when a planner-guessed
series_id is rejected by FRED with a permanent 400 ("series does not exist"),
the domain agent should recover via FRED's own discovery endpoints
(series/search, falling back to tags/series) before giving up."""
from unittest.mock import AsyncMock, patch

import pytest

from agents.macro_geopolitics_agent import MacroGeopoliticsAgent


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

    async def route(tool_name, args):
        if tool_name == "get_macro_indicator" and args.get("series_id") == "STEEL_PRODUCTION_INDEX":
            raise _permanent_series_error("STEEL_PRODUCTION_INDEX")
        if tool_name == "search_fred_series":
            return {"results": [{"series_id": "INDPRO", "title": "Industrial Production Index"}], "count": 1}
        if tool_name == "get_macro_indicator" and args.get("series_id") == "INDPRO":
            return good
        raise AssertionError(f"unexpected call: {tool_name}({args})")

    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=AsyncMock(side_effect=route)):
        agent = MacroGeopoliticsAgent()
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicator", {"series_id": "STEEL_PRODUCTION_INDEX"}
        )

    assert err is None
    assert result == good


@pytest.mark.asyncio
async def test_falls_back_to_tags_series_when_search_is_empty():
    """search_fred_series returns nothing; get_fred_series_by_tags recovers an id."""
    good = {"series_id": "WPU101", "title": "PPI: Steel", "observations": []}

    async def route(tool_name, args):
        if tool_name == "get_macro_indicator" and args.get("series_id") == "STEEL_PRICE_INDEX":
            raise _permanent_series_error("STEEL_PRICE_INDEX")
        if tool_name == "search_fred_series":
            return {"results": [], "count": 0}
        if tool_name == "get_fred_series_by_tags":
            assert args["tag_names"] == "steel;price;index"
            return {"results": [{"series_id": "WPU101", "title": "PPI: Steel"}], "count": 1}
        if tool_name == "get_macro_indicator" and args.get("series_id") == "WPU101":
            return good
        raise AssertionError(f"unexpected call: {tool_name}({args})")

    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=AsyncMock(side_effect=route)):
        agent = MacroGeopoliticsAgent()
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicator", {"series_id": "STEEL_PRICE_INDEX"}
        )

    assert err is None
    assert result == good


@pytest.mark.asyncio
async def test_gives_up_when_both_discovery_endpoints_are_empty():
    """Neither discovery endpoint finds a match — original failure is preserved."""

    async def route(tool_name, args):
        if tool_name == "get_macro_indicator":
            raise _permanent_series_error("NONSENSE_ID")
        if tool_name in ("search_fred_series", "get_fred_series_by_tags"):
            return {"results": [], "count": 0}
        raise AssertionError(f"unexpected call: {tool_name}({args})")

    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=AsyncMock(side_effect=route)):
        agent = MacroGeopoliticsAgent()
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicator", {"series_id": "NONSENSE_ID"}
        )

    assert result is None
    assert err is not None
    assert "NONSENSE_ID" in err


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
        agent = MacroGeopoliticsAgent()
        result, usages, err = await agent._call_tool_with_repair(
            "search_sec_filings", {"ticker": "CAT"}
        )

    assert result is None
    assert err is not None
    assert route.call_count == 1
