import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents import make_domain_agent
from core.schemas import ChapterDraft


def _make_plan(tool_names: list[str], domain: str = "competition") -> dict:
    return {
        "plan_id": "plan_001",
        "domain_activations": {domain: True},
        "tool_calls": [
            {"tool": name, "domain": domain, "arguments": {"query": "test"}}
            for name in tool_names
        ],
    }


def _crew_mock(domain: str = "competition") -> MagicMock:
    draft = {
        "domain": domain,
        "plan_id": "plan_001",
        "text": "Test chapter text.",
        "figures": {"CAT_revenue": "$64B"},
        "citations": ["https://example.com"],
        "contradiction_flags": [],
    }
    mock = MagicMock()
    mock.kickoff.return_value = MagicMock(__str__=lambda self: json.dumps(draft))
    return mock


@pytest.mark.asyncio
async def test_competition_agent_returns_chapter_draft():
    plan = _make_plan(["get_company_financials", "news_search"])
    tool_result = {"articles": [{"title": "CAT earnings", "description": "CAT revenue $64B"}]}

    with patch("agents.base_domain_agent.async_route", new=AsyncMock(return_value=tool_result)), \
         patch("agents.base_domain_agent.Crew", return_value=_crew_mock()), \
         patch("agents.base_domain_agent.LLM"):
        agent = make_domain_agent("competition")
        draft = await agent.run(plan, "run_001")

    assert isinstance(draft, ChapterDraft)
    assert draft.domain == "competition"
    assert draft.plan_id == "plan_001"
    assert draft.text


@pytest.mark.asyncio
async def test_competition_agent_fallback_on_crew_failure():
    plan = _make_plan(["get_company_financials"])
    tool_result = {"financials": {"revenue": 64_000_000_000}}

    crew_mock = MagicMock()
    crew_mock.kickoff.side_effect = RuntimeError("LLM timeout")

    with patch("agents.base_domain_agent.async_route", new=AsyncMock(return_value=tool_result)), \
         patch("agents.base_domain_agent.Crew", return_value=crew_mock), \
         patch("agents.base_domain_agent.LLM"):
        agent = make_domain_agent("competition")
        draft = await agent.run(plan, "run_001")

    assert isinstance(draft, ChapterDraft)
    assert draft.domain == "competition"
    assert draft.text  # fallback must produce non-empty text


@pytest.mark.asyncio
async def test_competition_agent_empty_plan_returns_placeholder():
    # Plan has no tool_calls for competition
    plan = {
        "plan_id": "plan_002",
        "domain_activations": {"commodities": True},
        "tool_calls": [
            {"tool": "get_mining_metals_prices", "domain": "commodities", "arguments": {}}
        ],
    }

    with patch("agents.base_domain_agent.LLM"):
        agent = make_domain_agent("competition")
        draft = await agent.run(plan, "run_001")

    assert isinstance(draft, ChapterDraft)
    assert "No tool calls" in draft.text
