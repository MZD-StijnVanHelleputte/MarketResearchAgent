import json
import pytest
from unittest.mock import MagicMock, patch
from core.tot.schemas import CandidatePlan
from agents.grounding_agent import GroundingAgent


def _make_plan(tool_names: list[str], plan_id: str = "plan_001") -> CandidatePlan:
    return CandidatePlan(
        plan_id=plan_id,
        domain_activations={"competition": True, "commodities": True},
        tool_calls=[
            {"tool": name, "domain": "competition", "arguments": {"query": "test"}}
            for name in tool_names
        ],
        rationale="Test plan",
        feasibility_score=0.0,
        quality_score=0.0,
        depth=1,
    )


def _grounded_response(plan: CandidatePlan, keep_tools: list[str]) -> dict:
    return {
        "plan_id": plan.plan_id,
        "domain_activations": plan.domain_activations,
        "entity_choices": {},
        "api_assignments": {},
        "tool_calls": [
            {"tool": t, "domain": "competition", "arguments": {"query": "test"}}
            for t in keep_tools
        ],
        "estimated_token_cost": 500,
        "rationale": plan.rationale,
        "depth": 2,
        "feasibility_score": 0.85,
        "quality_score": 0.80,
        "combined_score": 0.0,
        "gap_report": ["nonexistent_tool dropped: no data source available"],
        "receives_diversity_penalty": False,
        "is_survivor": False,
    }


def test_grounding_agent_drops_unavailable_tool():
    """A tool not in COLLECT_TOOLS registry must be removed and noted in gap_report."""
    plan = _make_plan(["news_search", "nonexistent_tool"])

    grounded_dict = _grounded_response(plan, keep_tools=["news_search"])
    mock_result = MagicMock()
    mock_result.__str__ = lambda self: json.dumps(grounded_dict)

    with patch("agents.grounding_agent.Crew") as MockCrew, \
         patch("agents.grounding_agent.LLM"):
        MockCrew.return_value.kickoff.return_value = mock_result
        agent = GroundingAgent()
        result = agent.run([plan])

    assert len(result) == 1
    grounded = result[0]
    assert grounded.depth == 2
    tool_names = [tc["tool"] for tc in grounded.tool_calls]
    assert "nonexistent_tool" not in tool_names
    assert "news_search" in tool_names
    assert len(grounded.gap_report) > 0


def test_grounding_agent_sets_feasibility_and_quality():
    plan = _make_plan(["news_search"])
    grounded_dict = _grounded_response(plan, keep_tools=["news_search"])
    mock_result = MagicMock()
    mock_result.__str__ = lambda self: json.dumps(grounded_dict)

    with patch("agents.grounding_agent.Crew") as MockCrew, \
         patch("agents.grounding_agent.LLM"):
        MockCrew.return_value.kickoff.return_value = mock_result
        result = GroundingAgent().run([plan])

    assert 0.0 <= result[0].feasibility_score <= 1.0
    assert 0.0 <= result[0].quality_score <= 1.0


def test_grounding_agent_fallback_on_llm_failure():
    """When CrewAI raises, the programmatic fallback should still drop unavailable tools."""
    plan = _make_plan(["news_search", "ghost_tool"])

    with patch("agents.grounding_agent.Crew") as MockCrew, \
         patch("agents.grounding_agent.LLM"):
        MockCrew.return_value.kickoff.side_effect = RuntimeError("LLM timeout")
        result = GroundingAgent().run([plan])

    assert len(result) == 1
    tool_names = [tc["tool"] for tc in result[0].tool_calls]
    assert "ghost_tool" not in tool_names
    assert "news_search" in tool_names
    assert any("ghost_tool" in msg for msg in result[0].gap_report)


def test_grounding_agent_applies_diversity_penalty():
    """Two plans with identical domain_activations: the second should get a penalty."""
    plan_a = _make_plan(["news_search"], plan_id="plan_001")
    plan_b = _make_plan(["web_search"], plan_id="plan_002")

    def make_mock_result(plan):
        d = _grounded_response(plan, keep_tools=plan.tool_calls[0]["tool"].split())
        d["feasibility_score"] = 0.9
        m = MagicMock()
        m.__str__ = lambda self, d=d: json.dumps(d)
        return m

    with patch("agents.grounding_agent.Crew") as MockCrew, \
         patch("agents.grounding_agent.LLM"):
        mock_crew = MockCrew.return_value
        mock_crew.kickoff.side_effect = [
            make_mock_result(plan_a),
            make_mock_result(plan_b),
        ]
        results = GroundingAgent().run([plan_a, plan_b])

    penalties = [p.receives_diversity_penalty for p in results]
    assert any(penalties), "At least one plan should receive diversity penalty"
