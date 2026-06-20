"""Round-trip tests for the FastMCP planning state bus."""
import pytest
from core.tot.schemas import CandidatePlan
from state_bus.server import (
    clear_plans,
    get_all_plans_direct,
    write_plan_direct,
    get_plans,
    get_coverage,
)


def _make_plan(plan_id: str, survivor: bool = False, active_domains: list[str] | None = None) -> CandidatePlan:
    activations = {
        "competition": "competition" in (active_domains or []),
        "commodities": "commodities" in (active_domains or []),
        "general_search": "general_search" in (active_domains or []),
        "distributors": False,
        "customers": False,
        "mining_projects": False,
        "macro_geopolitics": False,
    }
    return CandidatePlan(
        plan_id=plan_id,
        domain_activations=activations,
        tool_calls=[],
        feasibility_score=0.8,
        quality_score=0.7,
        is_survivor=survivor,
    )


def setup_function():
    clear_plans()


def test_write_and_read_plan():
    plan = _make_plan("plan_001")
    write_plan_direct(plan)
    all_plans = get_all_plans_direct()
    assert len(all_plans) == 1
    assert all_plans[0]["plan_id"] == "plan_001"


def test_round_trip_fidelity():
    plan = _make_plan("plan_abc", survivor=True, active_domains=["competition", "commodities"])
    write_plan_direct(plan)
    retrieved = get_all_plans_direct()[0]
    assert retrieved["feasibility_score"] == plan.feasibility_score
    assert retrieved["quality_score"] == plan.quality_score
    assert retrieved["is_survivor"] is True
    assert retrieved["domain_activations"]["competition"] is True
    assert retrieved["domain_activations"]["commodities"] is True


def test_upsert_overwrites_existing_plan():
    plan_v1 = _make_plan("plan_001")
    plan_v2 = _make_plan("plan_001", survivor=True)
    write_plan_direct(plan_v1)
    write_plan_direct(plan_v2)
    all_plans = get_all_plans_direct()
    assert len(all_plans) == 1
    assert all_plans[0]["is_survivor"] is True


def test_clear_removes_all_plans():
    write_plan_direct(_make_plan("plan_001"))
    write_plan_direct(_make_plan("plan_002"))
    clear_plans()
    assert get_all_plans_direct() == []


def test_get_plans_resource_returns_all():
    write_plan_direct(_make_plan("plan_001"))
    write_plan_direct(_make_plan("plan_002"))
    result = get_plans()
    assert "plans" in result
    assert len(result["plans"]) == 2


def test_coverage_resource_counts_active_survivor_domains():
    write_plan_direct(_make_plan("plan_001", survivor=True, active_domains=["competition"]))
    write_plan_direct(_make_plan("plan_002", survivor=True, active_domains=["competition", "commodities"]))
    write_plan_direct(_make_plan("plan_003", survivor=False, active_domains=["general_search"]))

    coverage = get_coverage()["coverage"]
    assert coverage.get("competition") == 2   # both survivors have it
    assert coverage.get("commodities") == 1
    assert coverage.get("general_search", 0) == 0  # non-survivor not counted


def test_write_multiple_plans_preserves_all():
    for i in range(7):
        write_plan_direct(_make_plan(f"plan_{i:03d}"))
    assert len(get_all_plans_direct()) == 7
