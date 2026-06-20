import json
import pytest
from unittest.mock import AsyncMock, patch
from models.response_parser import LLMResponse
from core.tot.proposer import PlanProposer
from core.tot.schemas import CandidatePlan

_DOMAIN_ACTIVATIONS = {
    "competition": True, "distributors": False, "customers": False,
    "mining_projects": False, "commodities": True, "macro_geopolitics": False,
    "general_search": False,
}

_REQUIRED_FIELDS = (
    "plan_id", "domain_activations", "tool_calls", "rationale",
    "feasibility_score", "quality_score", "depth",
)


def _make_raw_plan(i: int, activations: dict | None = None) -> dict:
    return {
        "plan_id": f"plan_{i:03d}",
        "domain_activations": activations or dict(_DOMAIN_ACTIVATIONS),
        "entity_choices": {},
        "api_assignments": {},
        "tool_calls": [{"tool": "news_search", "domain": "competition", "arguments": {"query": "test"}}],
        "estimated_token_cost": 500,
        "rationale": f"Test plan {i}",
        "depth": 1,
        "feasibility_score": 0.0,
        "quality_score": 0.0,
        "combined_score": 0.0,
        "gap_report": [],
        "receives_diversity_penalty": False,
        "is_survivor": False,
    }


def _make_seven_plans_json() -> str:
    plans = [_make_raw_plan(i + 1) for i in range(7)]
    return json.dumps({"plans": plans})


@pytest.mark.asyncio
async def test_proposer_returns_branching_factor_plans():
    mock_response = LLMResponse(content=_make_seven_plans_json())
    with patch("core.tot.proposer.LLMClient") as MockLLM:
        MockLLM.return_value.acomplete = AsyncMock(return_value=mock_response)
        candidates = await PlanProposer().propose("test query", [])

    assert len(candidates) == 7


@pytest.mark.asyncio
async def test_proposer_returns_candidate_plan_instances():
    mock_response = LLMResponse(content=_make_seven_plans_json())
    with patch("core.tot.proposer.LLMClient") as MockLLM:
        MockLLM.return_value.acomplete = AsyncMock(return_value=mock_response)
        candidates = await PlanProposer().propose("test", [])

    for plan in candidates:
        assert isinstance(plan, CandidatePlan)
        for field in _REQUIRED_FIELDS:
            assert hasattr(plan, field)


@pytest.mark.asyncio
async def test_proposer_sets_depth_1():
    mock_response = LLMResponse(content=_make_seven_plans_json())
    with patch("core.tot.proposer.LLMClient") as MockLLM:
        MockLLM.return_value.acomplete = AsyncMock(return_value=mock_response)
        candidates = await PlanProposer().propose("test", [])

    assert all(p.depth == 1 for p in candidates)


@pytest.mark.asyncio
async def test_proposer_raises_if_fewer_than_branching_factor():
    only_three = json.dumps({"plans": [_make_raw_plan(i + 1) for i in range(3)]})
    mock_response = LLMResponse(content=only_three)
    with patch("core.tot.proposer.LLMClient") as MockLLM:
        MockLLM.return_value.acomplete = AsyncMock(return_value=mock_response)
        with pytest.raises(ValueError, match="expected 7"):
            await PlanProposer().propose("test", [])


@pytest.mark.asyncio
async def test_proposer_uses_high_temperature():
    mock_response = LLMResponse(content=_make_seven_plans_json())
    with patch("core.tot.proposer.LLMClient") as MockLLM:
        instance = MockLLM.return_value
        instance.acomplete = AsyncMock(return_value=mock_response)
        await PlanProposer().propose("test", [])
        _, kwargs = instance.acomplete.call_args
        from config import settings
        assert kwargs.get("temperature") == settings.llm.propose_temperature


@pytest.mark.asyncio
async def test_proposer_uses_propose_max_tokens():
    mock_response = LLMResponse(content=_make_seven_plans_json())
    with patch("core.tot.proposer.LLMClient") as MockLLM:
        instance = MockLLM.return_value
        instance.acomplete = AsyncMock(return_value=mock_response)
        await PlanProposer().propose("test", [])
        _, kwargs = instance.acomplete.call_args
        from config import settings
        assert kwargs.get("max_tokens") == settings.llm.propose_max_tokens


@pytest.mark.asyncio
async def test_proposer_retries_once_on_truncated_json():
    truncated = '{"plans": [{"plan_id": "plan_001", "domain_activations": {'  # unterminated
    valid = _make_seven_plans_json()
    with patch("core.tot.proposer.LLMClient") as MockLLM:
        instance = MockLLM.return_value
        instance.acomplete = AsyncMock(side_effect=[
            LLMResponse(content=truncated),
            LLMResponse(content=valid),
        ])
        candidates = await PlanProposer().propose("test query", [])

    assert len(candidates) == 7
    assert instance.acomplete.call_count == 2


@pytest.mark.asyncio
async def test_proposer_raises_if_retry_also_fails():
    truncated = '{"plans": [{"plan_id": "plan_001"'
    with patch("core.tot.proposer.LLMClient") as MockLLM:
        instance = MockLLM.return_value
        instance.acomplete = AsyncMock(return_value=LLMResponse(content=truncated))
        with pytest.raises(json.JSONDecodeError):
            await PlanProposer().propose("test query", [])

    assert instance.acomplete.call_count == 2
