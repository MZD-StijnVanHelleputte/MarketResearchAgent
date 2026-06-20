import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.macro_geopolitics_agent import MacroGeopoliticsAgent
from core.schemas import ChapterDraft

_DOMAIN = "macro_geopolitics"


def _make_plan(tool_names: list[str]) -> dict:
    return {
        "plan_id": "plan_001",
        "domain_activations": {_DOMAIN: True},
        "tool_calls": [{"tool": n, "domain": _DOMAIN, "arguments": {}} for n in tool_names],
    }


def _crew_mock() -> MagicMock:
    draft = {"domain": _DOMAIN, "plan_id": "plan_001", "text": "Macro & geopolitics analysis.",
             "figures": {}, "citations": [], "contradiction_flags": []}
    m = MagicMock()
    m.kickoff.return_value = MagicMock(__str__=lambda self: json.dumps(draft))
    return m


@pytest.mark.asyncio
async def test_returns_chapter_draft():
    plan = _make_plan(["news_search", "web_search"])
    with patch("agents.base_domain_agent.async_route", new=AsyncMock(return_value={"articles": []})), \
         patch("agents.base_domain_agent.Crew", return_value=_crew_mock()), \
         patch("agents.base_domain_agent.LLM"):
        draft = await MacroGeopoliticsAgent().run(plan, "run_001")
    assert isinstance(draft, ChapterDraft)
    assert draft.domain == _DOMAIN


@pytest.mark.asyncio
async def test_fallback_on_crew_failure():
    plan = _make_plan(["web_search"])
    cm = MagicMock()
    cm.kickoff.side_effect = RuntimeError("fail")
    with patch("agents.base_domain_agent.async_route", new=AsyncMock(return_value={"results": [{"title": "t", "content": "c"}]})), \
         patch("agents.base_domain_agent.Crew", return_value=cm), \
         patch("agents.base_domain_agent.LLM"):
        draft = await MacroGeopoliticsAgent().run(plan, "run_001")
    assert isinstance(draft, ChapterDraft)
    assert draft.text


@pytest.mark.asyncio
async def test_empty_plan_returns_placeholder():
    plan = {"plan_id": "plan_x", "domain_activations": {}, "tool_calls": []}
    with patch("agents.base_domain_agent.LLM"):
        draft = await MacroGeopoliticsAgent().run(plan, "run_001")
    assert "No tool calls" in draft.text
