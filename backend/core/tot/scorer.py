"""Tree-of-Thought plan scorer and pruner."""
import logging

from config import settings
from core.tot.schemas import CandidatePlan

logger = logging.getLogger(__name__)


def score_and_prune(plans: list[CandidatePlan]) -> list[CandidatePlan]:
    """Compute combined_score for each plan, sort descending, mark top N survivors.

    Formula: combined_score = feasibility_weight * feasibility_score
                             + quality_weight    * quality_score
                             - diversity_penalty (if flagged)
    """
    fw = settings.tot.feasibility_weight
    qw = settings.tot.quality_weight
    dp = settings.tot.diversity_penalty
    n_survivors = settings.tot.survivors

    for plan in plans:
        penalty = dp if plan.receives_diversity_penalty else 0.0
        plan.combined_score = (
            fw * plan.feasibility_score
            + qw * plan.quality_score
            - penalty
        )

    plans.sort(key=lambda p: p.combined_score, reverse=True)

    for i, plan in enumerate(plans):
        plan.is_survivor = i < n_survivors

    survivors = [p for p in plans if p.is_survivor]
    logger.info(
        "score_and_prune: %d plans → %d survivors (top score %.3f)",
        len(plans),
        len(survivors),
        survivors[0].combined_score if survivors else 0.0,
    )
    return plans
