import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.commodities_agent import CommoditiesAgent
from core.schemas import ChapterDraft

_DOMAIN = "commodities"


def _make_plan(tool_names: list[str]) -> dict:
    return {
        "plan_id": "plan_001",
        "domain_activations": {_DOMAIN: True},
        "tool_calls": [
            {"tool": n, "domain": _DOMAIN, "arguments": {"symbol": "COPPER", "interval": "monthly"}}
            for n in tool_names
        ],
    }


def _crew_mock() -> MagicMock:
    draft = {"domain": _DOMAIN, "plan_id": "plan_001", "text": "Commodities analysis.",
             "figures": {"copper_spot_usd_per_lb": "4.12"}, "citations": [], "contradiction_flags": []}
    m = MagicMock()
    m.kickoff.return_value = MagicMock(__str__=lambda self: json.dumps(draft))
    return m


@pytest.mark.asyncio
async def test_returns_chapter_draft():
    plan = _make_plan(["get_mining_metals_prices", "get_broad_commodity_cycle"])
    with patch("agents.base_domain_agent.async_route", new=AsyncMock(return_value={"symbol": "COPPER", "latest": {"value": 4.12}})), \
         patch("agents.base_domain_agent.Agent"), \
         patch("agents.base_domain_agent.Task"), \
         patch("agents.base_domain_agent.Crew", return_value=_crew_mock()), \
         patch("agents.base_domain_agent.LLM"):
        draft = await CommoditiesAgent().run(plan, "run_001")
    assert isinstance(draft, ChapterDraft)
    assert draft.domain == _DOMAIN
    assert draft.figures.get("copper_spot_usd_per_lb") == "4.12"


@pytest.mark.asyncio
async def test_fallback_on_crew_failure():
    plan = _make_plan(["get_mining_metals_prices"])
    cm = MagicMock()
    cm.kickoff.side_effect = RuntimeError("fail")
    with patch("agents.base_domain_agent.async_route", new=AsyncMock(return_value={"symbol": "COPPER", "latest": {"value": 4.12}})), \
         patch("agents.base_domain_agent.Crew", return_value=cm), \
         patch("agents.base_domain_agent.LLM"):
        draft = await CommoditiesAgent().run(plan, "run_001")
    assert isinstance(draft, ChapterDraft)
    assert draft.text


@pytest.mark.asyncio
async def test_empty_plan_returns_placeholder():
    plan = {"plan_id": "plan_x", "domain_activations": {}, "tool_calls": []}
    with patch("agents.base_domain_agent.LLM"):
        draft = await CommoditiesAgent().run(plan, "run_001")
    assert "No tool calls" in draft.text
