import pytest
from config import settings
from core.tot.schemas import CandidatePlan
from core.tot.scorer import score_and_prune


def _make_plan(plan_id: str, feasibility: float, quality: float, penalty: bool = False) -> CandidatePlan:
    return CandidatePlan(
        plan_id=plan_id,
        domain_activations={"competition": True},
        feasibility_score=feasibility,
        quality_score=quality,
        receives_diversity_penalty=penalty,
    )


def _seven_plans(penalty_index: int | None = None) -> list[CandidatePlan]:
    data = [
        ("plan_001", 0.9, 0.8),
        ("plan_002", 0.8, 0.7),
        ("plan_003", 0.7, 0.9),
        ("plan_004", 0.6, 0.6),
        ("plan_005", 0.5, 0.5),
        ("plan_006", 0.4, 0.7),
        ("plan_007", 0.3, 0.4),
    ]
    return [
        _make_plan(pid, f, q, penalty=i == penalty_index)
        for i, (pid, f, q) in enumerate(data)
    ]


def test_exactly_three_survivors():
    plans = score_and_prune(_seven_plans())
    survivors = [p for p in plans if p.is_survivor]
    assert len(survivors) == settings.tot.survivors


def test_survivors_have_highest_combined_scores():
    plans = score_and_prune(_seven_plans())
    survivor_scores = sorted(
        [p.combined_score for p in plans if p.is_survivor], reverse=True
    )
    non_survivor_scores = [p.combined_score for p in plans if not p.is_survivor]
    assert min(survivor_scores) >= max(non_survivor_scores)


def test_score_formula_applied_correctly():
    fw = settings.tot.feasibility_weight
    qw = settings.tot.quality_weight
    plan = _make_plan("x", feasibility=0.8, quality=0.6)
    score_and_prune([plan])
    expected = fw * 0.8 + qw * 0.6
    assert abs(plan.combined_score - expected) < 1e-9


def test_diversity_penalty_subtracted():
    fw = settings.tot.feasibility_weight
    qw = settings.tot.quality_weight
    dp = settings.tot.diversity_penalty

    plan = _make_plan("x", feasibility=0.8, quality=0.6, penalty=True)
    score_and_prune([plan])
    expected = fw * 0.8 + qw * 0.6 - dp
    assert abs(plan.combined_score - expected) < 1e-9


def test_no_diversity_penalty_when_not_flagged():
    fw = settings.tot.feasibility_weight
    qw = settings.tot.quality_weight

    plan = _make_plan("x", feasibility=0.8, quality=0.6, penalty=False)
    score_and_prune([plan])
    expected = fw * 0.8 + qw * 0.6
    assert abs(plan.combined_score - expected) < 1e-9


def test_plans_sorted_descending():
    plans = score_and_prune(_seven_plans())
    scores = [p.combined_score for p in plans]
    assert scores == sorted(scores, reverse=True)


def test_diversity_penalised_plan_may_not_survive():
    # Give a plan high raw scores but flag it for diversity penalty
    plans = _seven_plans(penalty_index=0)
    result = score_and_prune(plans)
    # plan_001 had the highest raw scores; with penalty it may drop out of top-3
    # We only assert that the penalty lowers its score (tested above), not its rank
    penalised = next(p for p in result if p.receives_diversity_penalty)
    non_penalised_top = [p for p in result if not p.receives_diversity_penalty][:3]
    assert penalised.combined_score <= max(p.combined_score for p in non_penalised_top)
