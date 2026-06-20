"""Tests for adaptive tool-call repair and commodity figure preservation
in the base domain agent."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.commodities_agent import CommoditiesAgent


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
        agent = CommoditiesAgent()
        result, usages, err = await agent._call_tool_with_repair(
            "get_mining_metals_prices", {"symbol": "COPPER", "interval": "yearly"}
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
        agent = CommoditiesAgent()
        result, usages, err = await agent._call_tool_with_repair(
            "get_mining_metals_prices", {"symbol": "COPPER", "interval": "monthly"}
        )

    assert result is None
    assert err is not None
    # Only the original attempt — no blind retry of the same call.
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_fallback_preserves_commodity_figures():
    """A CommodityResult-shaped result lands in figures/text instead of being truncated."""
    with patch("agents.base_domain_agent.LLM"):
        agent = CommoditiesAgent()
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
