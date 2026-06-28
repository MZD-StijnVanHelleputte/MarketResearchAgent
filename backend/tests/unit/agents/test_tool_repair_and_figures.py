"""Tests for adaptive tool-call repair and commodity figure preservation
in the base domain agent."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents import make_domain_agent


def _llm_returning(payload: dict) -> MagicMock:
    """Mock LLMClient whose acomplete returns the given JSON decision."""
    client = MagicMock()
    client.acomplete = AsyncMock(
        return_value=SimpleNamespace(content=json.dumps(payload), usage={})
    )
    return client


@pytest.mark.asyncio
async def test_repair_adapts_arguments_and_retries():
    """A fixable error is interpreted by the LLM, args adapted, and the retry succeeds."""
    good = {"symbol": "COPPER", "unit": "USD/LB", "latest": {"date": "2024-12", "value": 4.12}}
    route = AsyncMock(side_effect=[
        Exception("Unsupported interval 'yearly' for COPPER. Use one of: annual, monthly, quarterly."),
        good,
    ])
    decision = {"action": "retry", "arguments": {"symbol": "COPPER", "interval": "monthly"},
                "reason": "fix interval"}
    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=route), \
         patch("agents.base_domain_agent.LLMClient", return_value=_llm_returning(decision)):
        agent = make_domain_agent("commodities")
        result, usages, err = await agent._call_tool_with_repair(
            "get_mining_metals_prices", {"symbol": "COPPER", "interval": "yearly"}, "c1"
        )

    assert err is None
    assert result == good
    assert route.call_count == 2
    # The retry used the adapted arguments.
    assert route.call_args.args[1] == {"symbol": "COPPER", "interval": "monthly"}


@pytest.mark.asyncio
async def test_repair_aborts_on_unfixable_error():
    """A quota error is not fixable by changing args → abort without re-calling the tool."""
    route = AsyncMock(side_effect=Exception(
        "Alpha Vantage did not return commodity data for 'COPPER': daily cap reached"
    ))
    decision = {"action": "abort", "reason": "rate limit"}
    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=route), \
         patch("agents.base_domain_agent.LLMClient", return_value=_llm_returning(decision)):
        agent = make_domain_agent("commodities")
        result, usages, err = await agent._call_tool_with_repair(
            "get_mining_metals_prices", {"symbol": "COPPER", "interval": "monthly"}, "c1"
        )

    assert result is None
    assert err is not None
    # Only the original attempt — no blind retry of the same call.
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_repair_caps_total_attempts_at_five():
    """A tool that always fails is attempted at most 5 times total (1 initial + 4 repairs)."""
    route = AsyncMock(side_effect=Exception("persistent upstream error"))
    # Each repair decision must propose new arguments (a repeat of the prior
    # arguments would make the repair loop break early as "no progress").
    decisions = [
        {"action": "retry", "arguments": {"symbol": "COPPER", "interval": f"variant_{i}"},
         "reason": "try a different interval"}
        for i in range(4)
    ]
    llm_client = MagicMock()
    llm_client.acomplete = AsyncMock(
        side_effect=[SimpleNamespace(content=json.dumps(d), usage={}) for d in decisions]
    )
    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=route), \
         patch("agents.base_domain_agent.LLMClient", return_value=llm_client):
        agent = make_domain_agent("commodities")
        result, usages, err = await agent._call_tool_with_repair(
            "get_mining_metals_prices", {"symbol": "COPPER", "interval": "monthly"}, "c1"
        )

    assert result is None
    assert err is not None
    assert route.call_count == 5


@pytest.mark.asyncio
async def test_repair_skips_loop_on_permanent_error():
    """A permanent HTTP 400 (e.g. FRED 'series does not exist') fails fast: the LLM
    repair loop is never entered, so no extra tool calls or LLM round-trips occur."""
    from clients.base_http_client import ClientError

    def _permanent_failure() -> Exception:
        try:
            try:
                raise ClientError(400, "Bad Request. The series does not exist.")
            except ClientError as inner:
                raise Exception("FRED observations failed for 'BADID'") from inner
        except Exception as exc:
            return exc

    route = AsyncMock(side_effect=_permanent_failure())
    llm_client = MagicMock()
    llm_client.acomplete = AsyncMock()  # must never be called
    with patch("agents.base_domain_agent.LLM"), \
         patch("agents.base_domain_agent.async_route", new=route), \
         patch("agents.base_domain_agent.LLMClient", return_value=llm_client):
        agent = make_domain_agent("commodities")
        result, usages, err = await agent._call_tool_with_repair(
            "get_macro_indicators", {"series_id": "BADID"}, "c1"
        )

    assert result is None
    assert err is not None
    assert route.call_count == 1          # no retry
    assert llm_client.acomplete.call_count == 0  # no LLM repair decision
    assert usages == []


@pytest.mark.asyncio
async def test_fallback_preserves_commodity_figures():
    """A CommodityResult-shaped result lands in figures/text instead of being truncated."""
    with patch("agents.base_domain_agent.LLM"):
        agent = make_domain_agent("commodities")
    raw = [{
        "tool": "get_mining_metals_prices",
        "result": {
            "symbol": "COPPER",
            "unit": "USD/LB",
            "latest": {"date": "2024-12-15", "value": 4.12},
            "rows": [{"date": "2024-11-15", "value": 4.08}],
        },
    }]
    draft = agent._fallback("plan_001", raw)

    assert any("4.12" in v for v in draft.figures.values())
    assert "4.12" in draft.text


def test_to_datasets_enriches_presentation_metadata():
    """Each normalized dataset carries data_type / label / count for the UI."""
    from agents.base_domain_agent import BaseDomainAgent

    raw = [
        {"tool": "get_mining_metals_prices", "result": {
            "symbol": "COPPER", "unit": "USD/LB",
            "rows": [{"date": "2024-11-15", "value": 4.08},
                     {"date": "2024-12-15", "value": 4.12}],
        }},
        {"tool": "news_search", "result": {
            "articles": [{"title": "CAT beats estimates", "url": "https://x/1", "description": "…"}],
        }},
        {"tool": "search_sec_filings", "result": {
            "filings": [{"entity_name": "Caterpillar Inc.", "form_type": "10-K",
                         "url": "https://sec/1", "file_date": "2025-02-01"}],
        }},
    ]
    datasets = BaseDomainAgent._to_datasets(raw)
    by_type = {d["data_type"]: d for d in datasets}

    assert by_type["numeric_series"]["label"] == "COPPER"
    assert by_type["numeric_series"]["count"] == 2
    assert by_type["articles"]["count"] == 1
    assert by_type["filings"]["label"] == "Caterpillar Inc."
    assert all("data_type" in d and "count" in d and "label" in d for d in datasets)
