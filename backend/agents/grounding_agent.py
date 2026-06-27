"""CrewAI-based depth-2 plan grounding agent.

For each depth-1 CandidatePlan the agent:
  1. Verifies every tool_call against the live COLLECT_TOOLS registry.
  2. Substitutes or drops tool references that don't exist, recording each
     change in gap_report.
  3. Assigns feasibility_score (0-1) and quality_score (0-1) based on how
     well the grounded plan covers the query.
  4. Sets depth=2 to indicate the plan has been grounded.

After all plans are grounded, applies a structural diversity check:
  - Plans whose domain_activations are identical to a higher-ranked plan
    (ranked by feasibility_score desc) receive receives_diversity_penalty=True.
"""
import concurrent.futures
import json
import logging

from crewai import Agent, Crew, LLM, Task

from config import settings
from core.tot.schemas import CandidatePlan
from models.usage import crew_usage, merge_usage
from tools.registry import COLLECT_TOOLS

logger = logging.getLogger(__name__)

_AVAILABLE_TOOLS = {t.name for t in COLLECT_TOOLS}

_GROUNDING_BACKSTORY = (
    "You are a rigorous research plan auditor for Komatsu's intelligence "
    "system. You review proposed research plans and rewrite them so they "
    "only reference tools that actually exist. You also score each plan on "
    "feasibility (can we actually collect this data?) and quality (will this "
    "data answer the query well?)."
)

_GROUNDING_TASK_TEMPLATE = """\
Review the following research plan and ground it against the available tools.

Available tools: {available_tools}

Plan to review:
{plan_json}

Instructions:
1. For each tool_call in the plan, check whether the tool name is in the \
available tools list.
2. If a tool is NOT available:
   - Try to substitute it with the closest available alternative.
   - If no substitute exists, remove the call and add an entry to gap_report \
explaining what data is missing.
3. Set feasibility_score (0.0-1.0): fraction of originally planned data \
sources that can actually be collected (after substitutions).
4. Set quality_score (0.0-1.0): estimate of how well the grounded plan will \
answer a Komatsu competitive-intelligence query given the available data. A plan \
that covers a named company with only financial/ticker calls (e.g. \
get_company_financials, get_equity_price) and no web_extract or news_search call \
for that company should score lower than an otherwise-similar plan that pairs \
financial calls with web_extract (company website/IR) and dated news_search \
coverage.
5. Set depth=2.

Respond with ONLY a valid JSON object (no markdown fences, no other text) \
matching the exact same schema as the input plan, with the following fields \
updated: tool_calls, gap_report, feasibility_score, quality_score, depth.
"""


class GroundingAgent:
    def __init__(self) -> None:
        # Accumulated token usage across all grounding calls (read by the graph).
        self.last_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
        self._llm = LLM(
            model=f"mistral/{settings.llm.model}",
            api_key=settings.mistral_api_key,
            temperature=settings.llm.work_temperature,
        )

    def run(self, plans: list[CandidatePlan]) -> list[CandidatePlan]:
        grounded: list[CandidatePlan] = []
        for plan in plans:
            try:
                grounded.append(self._ground_one(plan))
            except Exception as exc:
                logger.warning(
                    "GroundingAgent: failed to ground %s: %s — using fallback.",
                    plan.plan_id,
                    exc,
                )
                grounded.append(self._fallback_ground(plan))

        self._apply_diversity_penalties(grounded)
        logger.info("GroundingAgent: grounded %d plans.", len(grounded))
        return grounded

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ground_one(self, plan: CandidatePlan) -> CandidatePlan:
        agent = Agent(
            role="Plan Grounding Analyst",
            goal=(
                "Rewrite a research plan so it uses only available tools, "
                "score its feasibility and quality, and report any data gaps."
            ),
            backstory=_GROUNDING_BACKSTORY,
            llm=self._llm,
            verbose=False,
        )

        task = Task(
            description=_GROUNDING_TASK_TEMPLATE.format(
                available_tools=sorted(_AVAILABLE_TOOLS),
                plan_json=json.dumps(plan.model_dump(), indent=2),
            ),
            agent=agent,
            expected_output=(
                "A JSON object with the same schema as the input plan, "
                "fields tool_calls, gap_report, feasibility_score, "
                "quality_score, and depth updated."
            ),
        )

        crew = Crew(agents=[agent], tasks=[task], verbose=False)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(crew.kickoff).result()
        self.last_usage = merge_usage(self.last_usage, crew_usage(result))
        raw_output = str(result)

        updated = json.loads(raw_output)
        updated["plan_id"] = plan.plan_id
        updated["depth"] = 2
        updated.setdefault("domain_activations", plan.domain_activations)
        updated.setdefault("entity_choices", plan.entity_choices)
        updated.setdefault("api_assignments", plan.api_assignments)
        updated.setdefault("rationale", plan.rationale)
        updated.setdefault("estimated_token_cost", plan.estimated_token_cost)
        updated.setdefault("gap_report", [])
        updated.setdefault("receives_diversity_penalty", False)
        updated.setdefault("is_survivor", False)
        updated.setdefault("combined_score", 0.0)
        return CandidatePlan.model_validate(updated)

    def _fallback_ground(self, plan: CandidatePlan) -> CandidatePlan:
        """Programmatic fallback when the CrewAI call fails.

        Drops any tool_calls that reference unavailable tools and adds
        them to gap_report. Scores are conservative defaults.
        """
        valid_calls = []
        gap: list[str] = list(plan.gap_report)
        for tc in plan.tool_calls:
            if tc.get("tool") in _AVAILABLE_TOOLS:
                valid_calls.append(tc)
            else:
                gap.append(
                    f"Tool '{tc.get('tool')}' is not available; call dropped."
                )

        n_orig = len(plan.tool_calls)
        n_valid = len(valid_calls)
        feasibility = n_valid / n_orig if n_orig > 0 else 0.0

        return plan.model_copy(update={
            "tool_calls": valid_calls,
            "gap_report": gap,
            "feasibility_score": round(feasibility, 2),
            "quality_score": round(feasibility * 0.8, 2),
            "depth": 2,
        })

    def _apply_diversity_penalties(self, plans: list[CandidatePlan]) -> None:
        """Flag plans whose domain_activations are identical to a higher-feasibility peer."""
        sorted_plans = sorted(plans, key=lambda p: p.feasibility_score, reverse=True)
        seen_activations: list[frozenset] = []

        for plan in sorted_plans:
            activation_key = frozenset(
                d for d, v in plan.domain_activations.items() if v
            )
            if activation_key in seen_activations:
                plan.receives_diversity_penalty = True
            else:
                seen_activations.append(activation_key)
