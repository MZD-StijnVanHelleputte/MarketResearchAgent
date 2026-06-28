import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents import make_domain_agent
from core.schemas import ChapterDraft

_DOMAIN = "general_search"


def _make_plan(tool_names: list[str]) -> dict:
    return {
        "plan_id": "plan_001",
        "domain_activations": {_DOMAIN: True},
        "tool_calls": [{"tool": n, "domain": _DOMAIN, "arguments": {"query": "Komatsu"}} for n in tool_names],
    }


def _crew_mock() -> MagicMock:
    # "citations" here is deliberately ignored by the agent — citation identity is
    # always derived from raw tool results (see _extract_citations), never trusted
    # from the LLM's JSON response.
    draft = {"domain": _DOMAIN, "plan_id": "plan_001", "text": "General search analysis.",
             "figures": {}, "contradiction_flags": []}
    m = MagicMock()
    m.kickoff.return_value = MagicMock(__str__=lambda self: json.dumps(draft))
    return m


@pytest.mark.asyncio
async def test_returns_chapter_draft():
    plan = _make_plan(["web_search"])
    with patch("agents.base_domain_agent.async_route", new=AsyncMock(return_value={"results": [{"title": "t", "url": "https://example.com", "content": "c"}]})), \
         patch("agents.base_domain_agent.Agent"), \
         patch("agents.base_domain_agent.Task"), \
         patch("agents.base_domain_agent.Crew", return_value=_crew_mock()), \
         patch("agents.base_domain_agent.LLM"):
        draft = await make_domain_agent(_DOMAIN).run(plan, "run_001")
    assert isinstance(draft, ChapterDraft)
    assert draft.domain == _DOMAIN
    assert any(c.get("url") == "https://example.com" for c in draft.citations)


@pytest.mark.asyncio
async def test_fallback_on_crew_failure():
    plan = _make_plan(["web_search"])
    cm = MagicMock()
    cm.kickoff.side_effect = RuntimeError("fail")
    with patch("agents.base_domain_agent.async_route", new=AsyncMock(return_value={"results": [{"title": "t", "content": "c"}]})), \
         patch("agents.base_domain_agent.Crew", return_value=cm), \
         patch("agents.base_domain_agent.LLM"):
        draft = await make_domain_agent(_DOMAIN).run(plan, "run_001")
    assert isinstance(draft, ChapterDraft)
    assert draft.text


@pytest.mark.asyncio
async def test_empty_plan_returns_placeholder():
    plan = {"plan_id": "plan_x", "domain_activations": {}, "tool_calls": []}
    with patch("agents.base_domain_agent.LLM"):
        draft = await make_domain_agent(_DOMAIN).run(plan, "run_001")
    assert "No tool calls" in draft.text
