"""Tests for understand_node — now runs the full ToT pipeline.

Patches PlanProposer, GroundingAgent, and score_and_prune so the test
runs without real LLM calls or CrewAI.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.graph import understand_node, AgentState
from core.tot.schemas import CandidatePlan


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "run_id": "test-run",
        "session_id": "test-session",
        "user_query": "What is Caterpillar doing in autonomous trucks?",
        "plans": [],
        "collection_manifest": {},
        "chapter_sets": {},
        "synthesis_chapters": [],
        "merge_log": [],
        "confidence": 0.0,
        "react_iterations": 0,
        "context_messages": [],
        "warnings": [],
        "stage": "understand",
        "error": None,
        "cumulative_cost_usd": 0.0,
        "api_call_count": 0,
        "injection_flags": [],
        "clarification_done": True,  # skip clarification gate in unit tests
    }
    state.update(overrides)
    return state


def _make_candidate_plan(plan_id: str, survivor: bool = True) -> CandidatePlan:
    return CandidatePlan(
        plan_id=plan_id,
        domain_activations={
            "competition": True, "distributors": False, "customers": False,
            "mining_projects": False, "commodities": False,
            "macro_geopolitics": False, "general_search": True,
        },
        tool_calls=[{"tool": "news_search", "domain": "competition", "arguments": {"query": "test"}}],
        rationale="Test plan",
        feasibility_score=0.85,
        quality_score=0.80,
        combined_score=0.83,
        is_survivor=survivor,
    )


def _survivors(n: int = 3) -> list[CandidatePlan]:
    return [_make_candidate_plan(f"plan_{i:03d}", survivor=True) for i in range(1, n + 1)]


def _all_seven_plans() -> list[CandidatePlan]:
    return [_make_candidate_plan(f"plan_{i:03d}", survivor=i <= 3) for i in range(1, 8)]


@pytest.mark.asyncio
async def test_understand_node_happy_path():
    all_plans = _all_seven_plans()
    with patch("core.graph.PlanProposer") as MockProposer, \
         patch("core.graph.GroundingAgent") as MockGrounder, \
         patch("core.graph.score_and_prune", return_value=all_plans), \
         patch("core.graph.Retriever"), \
         patch("core.graph.write_plan_direct"), \
         patch("core.graph.clear_plans"):
        MockProposer.return_value.propose = AsyncMock(return_value=all_plans)
        MockGrounder.return_value.run = MagicMock(return_value=all_plans)
        result = await understand_node(_base_state())

    assert result["stage"] == "collect"
    assert len(result["plans"]) == 3


@pytest.mark.asyncio
async def test_understand_node_stores_survivor_dicts():
    all_plans = _all_seven_plans()
    with patch("core.graph.PlanProposer") as MockProposer, \
         patch("core.graph.GroundingAgent") as MockGrounder, \
         patch("core.graph.score_and_prune", return_value=all_plans), \
         patch("core.graph.Retriever"), \
         patch("core.graph.write_plan_direct"), \
         patch("core.graph.clear_plans"):
        MockProposer.return_value.propose = AsyncMock(return_value=all_plans)
        MockGrounder.return_value.run = MagicMock(return_value=all_plans)
        result = await understand_node(_base_state())

    for plan_dict in result["plans"]:
        assert isinstance(plan_dict, dict)
        assert "plan_id" in plan_dict
        assert plan_dict["is_survivor"] is True


@pytest.mark.asyncio
async def test_understand_node_writes_all_plans_to_mcp():
    all_plans = _all_seven_plans()
    with patch("core.graph.PlanProposer") as MockProposer, \
         patch("core.graph.GroundingAgent") as MockGrounder, \
         patch("core.graph.score_and_prune", return_value=all_plans), \
         patch("core.graph.Retriever"), \
         patch("core.graph.write_plan_direct") as mock_write, \
         patch("core.graph.clear_plans"):
        MockProposer.return_value.propose = AsyncMock(return_value=all_plans)
        MockGrounder.return_value.run = MagicMock(return_value=all_plans)
        await understand_node(_base_state())

    assert mock_write.call_count == 7


@pytest.mark.asyncio
async def test_understand_node_proposer_failure_returns_error():
    with patch("core.graph.PlanProposer") as MockProposer, \
         patch("core.graph.Retriever"), \
         patch("core.graph.clear_plans"):
        MockProposer.return_value.propose = AsyncMock(
            side_effect=ValueError("LLM returned bad JSON")
        )
        result = await understand_node(_base_state())

    assert result["stage"] == "error"
    assert "PlanProposer failed" in result["error"]


@pytest.mark.asyncio
async def test_understand_node_grounding_failure_falls_back():
    """If GroundingAgent fails, understand_node should use depth-1 plans and continue."""
    all_plans = _all_seven_plans()
    with patch("core.graph.PlanProposer") as MockProposer, \
         patch("core.graph.GroundingAgent") as MockGrounder, \
         patch("core.graph.score_and_prune", return_value=all_plans), \
         patch("core.graph.Retriever"), \
         patch("core.graph.write_plan_direct"), \
         patch("core.graph.clear_plans"):
        MockProposer.return_value.propose = AsyncMock(return_value=all_plans)
        MockGrounder.return_value.run = MagicMock(side_effect=RuntimeError("CrewAI timeout"))
        result = await understand_node(_base_state())

    assert result["stage"] == "collect"
    assert len(result["plans"]) == 3
