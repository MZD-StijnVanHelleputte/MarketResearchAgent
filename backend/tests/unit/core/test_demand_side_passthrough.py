"""Demand-side / consumer companies (BYD, VW, …) must flow through the pipeline.

They are neither equipment rivals nor mining operators, so they previously had no
field, no manifest entry, and no domain — and silently vanished from reports. These
tests pin the pass-through: research schema -> proposer prompt -> merged manifest.
"""
from core.tot.schemas import ResearchContext
from core.plan_merger import _fallback_merge
from prompts.propose_prompt import propose_messages


def _ctx() -> ResearchContext:
    return ResearchContext(
        competitors=["Caterpillar Inc. (CAT)"],
        operators=["BHP Group (BHP)"],
        demand_side_companies=["BYD (1211.HK)", "Volkswagen (VOW3.DE)"],
        tickers=["CAT", "BHP"],
        commodities=["Copper (HG=F)"],
    )


def test_research_context_has_demand_side_field():
    ctx = _ctx()
    assert ctx.demand_side_companies == ["BYD (1211.HK)", "Volkswagen (VOW3.DE)"]


def test_proposer_prompt_surfaces_demand_side_companies():
    messages = propose_messages("EV demand and copper", [], n=3, min_dims=3, research_context=_ctx())
    user = messages[1]["content"]
    system = messages[0]["content"]
    assert "BYD (1211.HK)" in user
    assert "Volkswagen (VOW3.DE)" in user
    # The system prompt must route demand-side consumers to general_search.
    assert "general_search" in system
    assert "Demand-side" in system or "demand-side" in system


def test_fallback_merge_carries_demand_side_into_manifest():
    consolidated = _fallback_merge([_survivor()], _ctx(), "run1")
    assert consolidated.entity_manifest["demand_side_companies"] == [
        "BYD (1211.HK)", "Volkswagen (VOW3.DE)",
    ]


def _survivor():
    from core.tot.schemas import CandidatePlan
    return CandidatePlan(
        plan_id="plan_001",
        domain_activations={"macro_geopolitics": True},
        tool_calls=[{"tool": "get_equity_price", "domain": "macro_geopolitics", "arguments": {"ticker": "BYDDY"}}],
    )
