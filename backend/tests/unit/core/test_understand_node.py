"""Tests for understand_node — now runs the full ToT pipeline.

Patches ResearchAgent, PlanProposer, GroundingAgent, score_and_prune, and
PlanMerger so the test runs without real LLM calls or CrewAI. The node now
merges the survivors into a single ConsolidatedPlan and returns plans=[that].
"""
import contextlib
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.graph import understand_node, AgentState
from core.tot.schemas import CandidatePlan, ConsolidatedPlan, ResearchContext


def _consolidated() -> ConsolidatedPlan:
    """The single merged plan understand_node now produces from the survivors."""
    return ConsolidatedPlan(
        plan_id="consolidated-test-run",
        source_plan_ids=["plan_001", "plan_002", "plan_003"],
        domains_active=["competition", "general_search"],
        entity_manifest={},
        planned_tool_calls=[],
        research_findings="",
        rationale="merged",
        gap_report="",
        feasibility_score=0.85,
        quality_score=0.80,
    )


@contextlib.contextmanager
def _patched_pipeline(all_plans, *, grounding_error=None):
    """Patch the whole ToT pipeline so understand_node runs offline and fast."""
    with patch("core.graph.ResearchAgent") as MockResearch, \
         patch("core.graph.PlanProposer") as MockProposer, \
         patch("core.graph.GroundingAgent") as MockGrounder, \
         patch("core.graph.score_and_prune", return_value=all_plans), \
         patch("core.graph.PlanMerger") as MockMerger, \
         patch("core.graph.Retriever"), \
         patch("core.graph.write_plan_direct") as mock_write, \
         patch("core.graph.clear_plans"):
        MockResearch.return_value.run = AsyncMock(return_value=ResearchContext())
        MockResearch.return_value.last_usage = {}
        MockProposer.return_value.propose = AsyncMock(return_value=all_plans)
        MockProposer.return_value.last_usage = {}
        if grounding_error is not None:
            MockGrounder.return_value.run = MagicMock(side_effect=grounding_error)
        else:
            MockGrounder.return_value.run = MagicMock(return_value=all_plans)
        MockGrounder.return_value.last_usage = {}
        MockMerger.return_value.merge = AsyncMock(return_value=_consolidated())
        MockMerger.return_value.last_usage = {}
        yield mock_write


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
    with _patched_pipeline(all_plans):
        result = await understand_node(_base_state())

    assert result["stage"] == "collect"
    # Survivors are merged into a single consolidated plan.
    assert len(result["plans"]) == 1
    assert result["plans"][0]["plan_id"] == "consolidated-test-run"


@pytest.mark.asyncio
async def test_understand_node_stores_consolidated_plan_dict():
    all_plans = _all_seven_plans()
    with _patched_pipeline(all_plans):
        result = await understand_node(_base_state())

    assert len(result["plans"]) == 1
    plan_dict = result["plans"][0]
    assert isinstance(plan_dict, dict)
    assert "plan_id" in plan_dict
    # The merged plan records which survivors fed it.
    assert plan_dict["source_plan_ids"] == ["plan_001", "plan_002", "plan_003"]


@pytest.mark.asyncio
async def test_understand_node_writes_all_plans_to_mcp():
    all_plans = _all_seven_plans()
    with _patched_pipeline(all_plans) as mock_write:
        await understand_node(_base_state())

    assert mock_write.call_count == 7


@pytest.mark.asyncio
async def test_understand_node_proposer_failure_returns_error():
    all_plans = _all_seven_plans()
    with _patched_pipeline(all_plans), \
         patch("core.graph.PlanProposer") as MockProposer:
        MockProposer.return_value.propose = AsyncMock(
            side_effect=ValueError("LLM returned bad JSON")
        )
        result = await understand_node(_base_state())

    assert result["stage"] == "error"
    assert "PlanProposer failed" in result["error"]


@pytest.mark.asyncio
async def test_understand_node_grounding_failure_falls_back():
    """If GroundingAgent fails, understand_node uses depth-1 plans and still merges."""
    all_plans = _all_seven_plans()
    with _patched_pipeline(all_plans, grounding_error=RuntimeError("CrewAI timeout")):
        result = await understand_node(_base_state())

    assert result["stage"] == "collect"
    assert len(result["plans"]) == 1


@pytest.mark.asyncio
async def test_understand_node_retries_grounding_when_gaps_present():
    """A consolidated plan with gap_report triggers one remediation round (default max=1)."""
    all_plans = _all_seven_plans()
    gapped = _consolidated()
    gapped.gap_report = "Tool 'x' unavailable; no substitute found."
    clean = _consolidated()

    with patch("core.graph.ResearchAgent") as MockResearch, \
         patch("core.graph.PlanProposer") as MockProposer, \
         patch("core.graph.GroundingAgent") as MockGrounder, \
         patch("core.graph.score_and_prune", return_value=all_plans), \
         patch("core.graph.PlanMerger") as MockMerger, \
         patch("core.graph.Retriever"), \
         patch("core.graph.write_plan_direct"), \
         patch("core.graph.clear_plans"), \
         patch("core.graph._timeout_interrupt_if_needed") as mock_timeout:
        MockResearch.return_value.run = AsyncMock(return_value=ResearchContext())
        MockResearch.return_value.last_usage = {}
        MockProposer.return_value.propose = AsyncMock(return_value=all_plans)
        MockProposer.return_value.last_usage = {}
        MockGrounder.return_value.run = MagicMock(return_value=all_plans)
        MockGrounder.return_value.last_usage = {}
        MockMerger.return_value.merge = AsyncMock(side_effect=[gapped, clean])
        MockMerger.return_value.last_usage = {}

        result = await understand_node(_base_state())

    assert result["stage"] == "collect"
    assert MockMerger.return_value.merge.await_count == 2
    assert MockGrounder.return_value.run.call_count == 2
    # Called once at node entry, once more for the single remediation round.
    assert mock_timeout.call_count == 2


@pytest.mark.asyncio
async def test_understand_node_stops_remediation_at_max_rounds():
    """Gaps that never clear stop after gap_remediation_max_rounds, not forever."""
    all_plans = _all_seven_plans()
    gapped = _consolidated()
    gapped.gap_report = "Persistent structural gap."

    with patch("core.graph.ResearchAgent") as MockResearch, \
         patch("core.graph.PlanProposer") as MockProposer, \
         patch("core.graph.GroundingAgent") as MockGrounder, \
         patch("core.graph.score_and_prune", return_value=all_plans), \
         patch("core.graph.PlanMerger") as MockMerger, \
         patch("core.graph.Retriever"), \
         patch("core.graph.write_plan_direct"), \
         patch("core.graph.clear_plans"):
        MockResearch.return_value.run = AsyncMock(return_value=ResearchContext())
        MockResearch.return_value.last_usage = {}
        MockProposer.return_value.propose = AsyncMock(return_value=all_plans)
        MockProposer.return_value.last_usage = {}
        MockGrounder.return_value.run = MagicMock(return_value=all_plans)
        MockGrounder.return_value.last_usage = {}
        MockMerger.return_value.merge = AsyncMock(return_value=gapped)
        MockMerger.return_value.last_usage = {}

        result = await understand_node(_base_state())

    assert result["stage"] == "collect"
    # Default gap_remediation_max_rounds=1: initial pass + 1 retry = 2 merge calls.
    assert MockMerger.return_value.merge.await_count == 2
