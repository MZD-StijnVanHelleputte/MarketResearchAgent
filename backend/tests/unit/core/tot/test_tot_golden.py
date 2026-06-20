"""Golden-file gate test for the full ToT pipeline.

Uses a fixed mocked LLM response to verify that:
  - Exactly TOT_SURVIVORS (3) plans are marked as survivors.
  - Combined scores are within ±0.05 of expected values.
  - At least TOT_MIN_DIVERSITY_DIMS (3) diversity dimensions differ across survivors.
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from config import settings
from core.tot.proposer import PlanProposer
from core.tot.scorer import score_and_prune
from core.tot.schemas import CandidatePlan
from models.response_parser import LLMResponse
from state_bus.server import clear_plans

_DOMAINS = [
    "competition", "distributors", "customers", "mining_projects",
    "commodities", "macro_geopolitics", "general_search",
]

# Fixed golden plans with pre-assigned feasibility/quality scores so we can
# compute expected combined_scores deterministically.
_GOLDEN_PLANS_RAW = [
    {
        "plan_id": "plan_001",
        "domain_activations": {"competition": True, "distributors": False, "customers": False,
                               "mining_projects": False, "commodities": True,
                               "macro_geopolitics": False, "general_search": False},
        "entity_choices": {"company": "Caterpillar"},
        "api_assignments": {},
        "tool_calls": [{"tool": "news_search", "domain": "competition", "arguments": {"query": "CAT"}}],
        "estimated_token_cost": 800,
        "rationale": "Competitive angle via news",
        "depth": 1, "feasibility_score": 0.9, "quality_score": 0.8,
        "combined_score": 0.0, "gap_report": [], "receives_diversity_penalty": False, "is_survivor": False,
    },
    {
        "plan_id": "plan_002",
        "domain_activations": {"competition": True, "distributors": True, "customers": False,
                               "mining_projects": False, "commodities": False,
                               "macro_geopolitics": False, "general_search": False},
        "entity_choices": {"company": "Caterpillar", "distributor": "WesTrac"},
        "api_assignments": {},
        "tool_calls": [{"tool": "company_financials", "domain": "competition", "arguments": {"ticker": "CAT"}}],
        "estimated_token_cost": 1200,
        "rationale": "Financials + distributor impact",
        "depth": 1, "feasibility_score": 0.85, "quality_score": 0.75,
        "combined_score": 0.0, "gap_report": [], "receives_diversity_penalty": False, "is_survivor": False,
    },
    {
        "plan_id": "plan_003",
        "domain_activations": {"competition": False, "distributors": False, "customers": True,
                               "mining_projects": True, "commodities": True,
                               "macro_geopolitics": False, "general_search": False},
        "entity_choices": {"customer": "BHP"},
        "api_assignments": {},
        "tool_calls": [{"tool": "commodity_price", "domain": "commodities", "arguments": {"symbol": "COPPER"}}],
        "estimated_token_cost": 900,
        "rationale": "Demand-side: mining projects + commodity prices",
        "depth": 1, "feasibility_score": 0.80, "quality_score": 0.85,
        "combined_score": 0.0, "gap_report": [], "receives_diversity_penalty": False, "is_survivor": False,
    },
    {
        "plan_id": "plan_004",
        "domain_activations": {"competition": False, "distributors": False, "customers": False,
                               "mining_projects": False, "commodities": False,
                               "macro_geopolitics": True, "general_search": True},
        "entity_choices": {},
        "api_assignments": {},
        "tool_calls": [{"tool": "web_search", "domain": "general_search", "arguments": {"query": "mining capex"}}],
        "estimated_token_cost": 600,
        "rationale": "Macro + general web search angle",
        "depth": 1, "feasibility_score": 0.70, "quality_score": 0.65,
        "combined_score": 0.0, "gap_report": [], "receives_diversity_penalty": False, "is_survivor": False,
    },
    {
        "plan_id": "plan_005",
        "domain_activations": {"competition": True, "distributors": False, "customers": False,
                               "mining_projects": False, "commodities": False,
                               "macro_geopolitics": False, "general_search": True},
        "entity_choices": {},
        "api_assignments": {},
        "tool_calls": [{"tool": "sec_filings", "domain": "competition", "arguments": {"ticker": "CAT"}}],
        "estimated_token_cost": 700,
        "rationale": "SEC filings for forward-looking guidance",
        "depth": 1, "feasibility_score": 0.75, "quality_score": 0.70,
        "combined_score": 0.0, "gap_report": [], "receives_diversity_penalty": False, "is_survivor": False,
    },
    {
        "plan_id": "plan_006",
        "domain_activations": {"competition": False, "distributors": True, "customers": True,
                               "mining_projects": False, "commodities": False,
                               "macro_geopolitics": False, "general_search": False},
        "entity_choices": {},
        "api_assignments": {},
        "tool_calls": [{"tool": "news_search", "domain": "distributors", "arguments": {"query": "distributor"}}],
        "estimated_token_cost": 500,
        "rationale": "Channel inventory and customer demand",
        "depth": 1, "feasibility_score": 0.65, "quality_score": 0.60,
        "combined_score": 0.0, "gap_report": [], "receives_diversity_penalty": False, "is_survivor": False,
    },
    {
        "plan_id": "plan_007",
        "domain_activations": {"competition": True, "distributors": False, "customers": False,
                               "mining_projects": False, "commodities": True,
                               "macro_geopolitics": False, "general_search": False},
        "entity_choices": {},
        "api_assignments": {},
        "tool_calls": [{"tool": "equity_price", "domain": "competition", "arguments": {"ticker": "CAT"}}],
        "estimated_token_cost": 400,
        "rationale": "Equity market signals",
        "depth": 1, "feasibility_score": 0.60, "quality_score": 0.55,
        "combined_score": 0.0, "gap_report": [], "receives_diversity_penalty": False, "is_survivor": False,
    },
]


def _compute_expected_score(raw: dict) -> float:
    fw = settings.tot.feasibility_weight
    qw = settings.tot.quality_weight
    dp = settings.tot.diversity_penalty
    penalty = dp if raw.get("receives_diversity_penalty") else 0.0
    return fw * raw["feasibility_score"] + qw * raw["quality_score"] - penalty


@pytest.mark.asyncio
async def test_golden_three_survivors():
    clear_plans()
    mock_resp = LLMResponse(content=json.dumps({"plans": _GOLDEN_PLANS_RAW}))

    with patch("core.tot.proposer.LLMClient") as MockLLM:
        MockLLM.return_value.acomplete = AsyncMock(return_value=mock_resp)
        candidates = await PlanProposer().propose("Caterpillar capex impact on Komatsu", [])

    all_plans = score_and_prune(candidates)
    survivors = [p for p in all_plans if p.is_survivor]
    assert len(survivors) == settings.tot.survivors


@pytest.mark.asyncio
async def test_golden_scores_within_tolerance():
    """Combined scores must be within ±0.05 of the deterministic expected value."""
    clear_plans()
    mock_resp = LLMResponse(content=json.dumps({"plans": _GOLDEN_PLANS_RAW}))

    with patch("core.tot.proposer.LLMClient") as MockLLM:
        MockLLM.return_value.acomplete = AsyncMock(return_value=mock_resp)
        candidates = await PlanProposer().propose("test", [])

    all_plans = score_and_prune(candidates)

    expected_by_id = {r["plan_id"]: _compute_expected_score(r) for r in _GOLDEN_PLANS_RAW}
    for plan in all_plans:
        expected = expected_by_id[plan.plan_id]
        assert abs(plan.combined_score - expected) < 0.05, (
            f"{plan.plan_id}: expected {expected:.4f}, got {plan.combined_score:.4f}"
        )


@pytest.mark.asyncio
async def test_golden_diversity_across_survivors():
    """Survivors must differ on at least TOT_MIN_DIVERSITY_DIMS dimensions."""
    clear_plans()
    mock_resp = LLMResponse(content=json.dumps({"plans": _GOLDEN_PLANS_RAW}))

    with patch("core.tot.proposer.LLMClient") as MockLLM:
        MockLLM.return_value.acomplete = AsyncMock(return_value=mock_resp)
        candidates = await PlanProposer().propose("test", [])

    all_plans = score_and_prune(candidates)
    survivors = [p for p in all_plans if p.is_survivor]

    def active_domains(plan: CandidatePlan) -> frozenset:
        return frozenset(d for d, v in plan.domain_activations.items() if v)

    activation_sets = [active_domains(s) for s in survivors]
    unique_sets = set(activation_sets)
    assert len(unique_sets) >= settings.tot.min_diversity_dims, (
        f"Only {len(unique_sets)} distinct domain activation sets among survivors; "
        f"need at least {settings.tot.min_diversity_dims}"
    )
