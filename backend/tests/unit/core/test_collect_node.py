"""Unit tests for collect_node (Phase 7: domain sub-agent dispatch).

Phase 7 replaced the flat tool loop with per-domain CrewAI sub-agents.
Tests patch core.graph._run_domain_agent to isolate collect_node logic.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.graph import collect_node, AgentState
from core.schemas import ChapterDraft


PLAN = {
    "plan_id": "plan_test",
    "feasibility_score": 0.8,
    "domain_activations": {
        "competition": True,
        "distributors": False,
        "customers": False,
        "mining_projects": False,
        "commodities": True,
        "macro_geopolitics": False,
        "general_search": False,
    },
    "tool_calls": [
        {"tool": "news_search", "domain": "competition", "arguments": {"query": "CAT"}},
        {"tool": "news_search", "domain": "commodities", "arguments": {"query": "copper"}},
    ],
    "rationale": "Test plan",
}


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "run_id": "test-run",
        "session_id": "test-session",
        "user_query": "test query",
        "plans": [PLAN],
        "collection_manifest": {},
        "chapter_sets": {},
        "synthesis_chapters": [],
        "merge_log": [],
        "confidence": 0.0,
        "react_iterations": 0,
        "context_messages": [],
        "warnings": [],
        "stage": "collect",
        "error": None,
    }
    state.update(overrides)
    return state


def _draft(domain: str, plan_id: str = "plan_test") -> ChapterDraft:
    return ChapterDraft(
        domain=domain,
        plan_id=plan_id,
        text=f"Chapter text for {domain}",
        figures={f"{domain}_metric": "100"},
    )


async def _fake_run_domain_agent(plan, domain, run_id, retriever, chunker, collection):
    return plan.get("plan_id", ""), domain, _draft(domain, plan.get("plan_id", ""))


@pytest.mark.asyncio
async def test_collect_node_all_succeed():
    with patch("core.graph._run_domain_agent", side_effect=_fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state())

    assert result["confidence"] == 1.0
    assert result["stage"] == "collect"
    assert "plan_test::competition" in result["chapter_sets"]
    assert "plan_test::commodities" in result["chapter_sets"]


@pytest.mark.asyncio
async def test_collect_node_one_fails():
    call_count = 0

    async def flaky_agent(plan, domain, run_id, retriever, chunker, collection):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("agent crash")
        return plan.get("plan_id", ""), domain, _draft(domain)

    with patch("core.graph._run_domain_agent", side_effect=flaky_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state())

    assert result["confidence"] == 0.5


@pytest.mark.asyncio
async def test_collect_node_no_plan():
    result = await collect_node(_base_state(plans=[]))
    assert result["stage"] == "error"
    assert result["error"] is not None


@pytest.mark.asyncio
async def test_collect_node_all_fail_confidence_zero():
    async def always_fail(plan, domain, *args, **kwargs):
        raise RuntimeError("fail")

    with patch("core.graph._run_domain_agent", side_effect=always_fail), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state())

    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_collect_node_populates_chapter_sets():
    with patch("core.graph._run_domain_agent", side_effect=_fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state())

    key = "plan_test::competition"
    assert key in result["chapter_sets"]
    draft_dict = result["chapter_sets"][key]
    assert draft_dict["domain"] == "competition"
    assert draft_dict["text"]


@pytest.mark.asyncio
async def test_collect_node_no_active_domains():
    """A plan with all domains False should produce an error."""
    empty_plan = dict(PLAN)
    empty_plan["domain_activations"] = {d: False for d in PLAN["domain_activations"]}
    with patch("core.graph._run_domain_agent", side_effect=_fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state(plans=[empty_plan]))

    assert result["stage"] == "error"


@pytest.mark.asyncio
async def test_collect_node_reuses_prior_successful_draft():
    """On a backtrack, only domains without a good prior draft are re-run."""
    ran: list[str] = []

    async def tracking_agent(plan, domain, run_id, retriever, chunker, collection):
        ran.append(domain)
        return plan.get("plan_id", ""), domain, _draft(domain, plan.get("plan_id", ""))

    # competition already has a good draft from a previous pass; commodities does not.
    prior = {
        "plan_test::competition": _draft("competition").model_dump(),
    }
    state = _base_state(chapter_sets=prior)
    with patch("core.graph._run_domain_agent", side_effect=tracking_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(state)

    assert ran == ["commodities"]  # competition reused, not re-run
    assert result["confidence"] == 1.0
    assert "plan_test::competition" in result["chapter_sets"]
    assert "plan_test::commodities" in result["chapter_sets"]


@pytest.mark.asyncio
async def test_collect_node_consolidated_plan_shape():
    """A ConsolidatedPlan dict (domains_active: list) must activate domains."""
    consolidated_plan = {
        "plan_id": "consolidated_test",
        "domains_active": ["competition", "commodities"],
        "planned_tool_calls": [
            {"tool": "news_search", "domain": "competition", "params": {"query": "CAT"}},
            {"tool": "news_search", "domain": "commodities", "params": {"query": "copper"}},
        ],
        "rationale": "Test consolidated plan",
    }
    with patch("core.graph._run_domain_agent", side_effect=_fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state(plans=[consolidated_plan]))

    assert result["stage"] != "error"
    assert result["confidence"] == 1.0
    assert "consolidated_test::competition" in result["chapter_sets"]
    assert "consolidated_test::commodities" in result["chapter_sets"]
