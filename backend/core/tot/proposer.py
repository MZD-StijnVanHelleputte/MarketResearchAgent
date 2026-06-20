"""Tree-of-Thought depth-1 plan proposer."""
from __future__ import annotations

import json
import logging

from config import settings
from models.llm_client import LLMClient
from models.usage import llm_usage
from models.response_parser import parse_plan_list
from prompts.propose_prompt import propose_messages
from retrieval.chroma_store import Chunk
from core.tot.schemas import CandidatePlan, ResearchContext

logger = logging.getLogger(__name__)


class PlanProposer:
    def __init__(self) -> None:
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}

    async def propose(
        self,
        query: str,
        context_chunks: list[Chunk],
        research_context: ResearchContext | None = None,
    ) -> list[CandidatePlan]:
        n = settings.tot.branching_factor
        min_dims = settings.tot.min_diversity_dims
        messages = propose_messages(query, context_chunks, n, min_dims, research_context)

        llm = LLMClient()
        response = await llm.acomplete(
            messages,
            temperature=settings.llm.propose_temperature,
            max_tokens=settings.llm.propose_max_tokens,
        )
        self.last_usage = llm_usage(response.usage)

        logger.debug("PlanProposer: raw LLM response: %.500s", response.content or "")
        try:
            raw_plans = parse_plan_list(response.content or "")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "PlanProposer: failed to parse plan list (%s) — retrying once. Raw: %.300s",
                exc, response.content or "",
            )
            response = await llm.acomplete(
                messages,
                temperature=settings.llm.propose_temperature,
                max_tokens=settings.llm.propose_max_tokens,
            )
            retry_usage = llm_usage(response.usage)
            self.last_usage = {
                k: self.last_usage.get(k, 0) + retry_usage.get(k, 0)
                for k in set(self.last_usage) | set(retry_usage)
            }
            try:
                raw_plans = parse_plan_list(response.content or "")
            except (json.JSONDecodeError, ValueError) as exc2:
                logger.error(
                    "PlanProposer: failed to parse plan list on retry — %s. Raw: %.300s",
                    exc2, response.content or "",
                )
                raise

        candidates: list[CandidatePlan] = []
        for i, raw in enumerate(raw_plans):
            raw["plan_id"] = raw.get("plan_id") or f"plan_{i + 1:03d}"
            raw["depth"] = 1
            raw.setdefault("is_survivor", False)
            raw.setdefault("receives_diversity_penalty", False)
            raw.setdefault("gap_report", [])
            raw.setdefault("feasibility_score", 0.0)
            raw.setdefault("quality_score", 0.0)
            raw.setdefault("combined_score", 0.0)
            candidates.append(CandidatePlan.model_validate(raw))

        if len(candidates) == 0:
            raise ValueError("Proposer returned 0 plans.")
        if len(candidates) < n:
            logger.warning(
                "PlanProposer: expected %d plans, got %d — proceeding with fewer.", n, len(candidates)
            )

        logger.info("PlanProposer: produced %d depth-1 candidates.", len(candidates))
        return candidates
