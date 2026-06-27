"""Merges the top-N ToT survivor plans into a single ConsolidatedPlan for Gate 1.

Instead of asking the user to pick one of three plans, the merger synthesises
the survivors into one comprehensive plan that takes the best coverage from each.
"""
from __future__ import annotations

import json
import logging
import re

from config import settings
from core.tot.schemas import (
    CandidatePlan,
    ConsolidatedPlan,
    DATASET_OWNERSHIP_PRIORITY,
    PlannedToolCall,
    ResearchContext,
)
from models.llm_client import LLMClient
from prompts.plan_merge_prompt import plan_merge_messages

logger = logging.getLogger(__name__)

# Tool names whose schema takes a single `ticker` string argument — these are
# expanded to one call per tracked ticker so coverage doesn't depend on the
# LLM planner enumerating every competitor itself. Premium-gated names are
# harmless to include; tools/registry.py's tier filter drops them later.
TICKER_PARAM_TOOLS: frozenset[str] = frozenset({
    "get_equity_price", "get_equity_history", "get_equity_financials",
    "get_company_financials", "get_income_statement", "get_balance_sheet",
    "get_cash_flow", "get_financial_ratios", "get_analyst_estimates",
    "get_stock_peers", "get_company_rating", "get_earnings_surprises",
    "get_press_releases", "get_earnings_calendar", "get_earnings_transcript",
    "get_news_sentiment", "get_insider_transactions",
})


def _expand_per_ticker_calls(
    planned_calls: list[PlannedToolCall],
    entity_manifest: dict,
) -> list[PlannedToolCall]:
    """Ensure every ticker-scoped tool already planned is called for every known
    ticker, not just the one(s) the planner happened to pick.

    For each tool name in TICKER_PARAM_TOOLS that appears at least once in
    planned_calls, clones the first matching call for any ticker in
    entity_manifest["tickers"] not already covered. Caps the resulting list at
    settings.safety.max_api_calls_per_run, favoring breadth over depth.
    """
    tickers: list[str] = entity_manifest.get("tickers") or []
    if not tickers:
        return planned_calls

    templates: dict[str, PlannedToolCall] = {}
    covered: dict[str, set[str]] = {}
    for tc in planned_calls:
        if tc.tool not in TICKER_PARAM_TOOLS:
            continue
        ticker = tc.params.get("ticker")
        if ticker is None:
            continue
        templates.setdefault(tc.tool, tc)
        covered.setdefault(tc.tool, set()).add(str(ticker).upper())

    expanded = list(planned_calls)
    for tool_name, template in templates.items():
        for ticker in tickers:
            if ticker.upper() in covered.get(tool_name, set()):
                continue
            expanded.append(PlannedToolCall(
                tool=tool_name,
                params={**template.params, "ticker": ticker},
                domain=template.domain,
                rationale=template.rationale,
            ))
            covered[tool_name].add(ticker.upper())

    max_calls = settings.safety.max_api_calls_per_run
    if len(expanded) > max_calls:
        logger.warning(
            "PlanMerger: per-ticker expansion produced %d calls, truncating to "
            "max_api_calls_per_run=%d", len(expanded), max_calls,
        )
        expanded = expanded[:max_calls]
    return expanded


_OWNERSHIP_RANK = {d: i for i, d in enumerate(DATASET_OWNERSHIP_PRIORITY)}


def _dedupe_cross_domain(calls: list[PlannedToolCall]) -> list[PlannedToolCall]:
    """Collapse identical tool+params calls that were assigned to more than one
    domain, keeping only the highest-priority domain's copy (DATASET_OWNERSHIP_PRIORITY).

    Without this, e.g. both "commodities" and "macro_geopolitics" can independently
    call get_mining_metals_prices(COPPER) for the same plan, doubling the API call
    and producing the same data in two report chapters.
    """
    best: dict[str, PlannedToolCall] = {}
    for tc in calls:
        key = f"{tc.tool}:{json.dumps(tc.params, sort_keys=True, default=str)}"
        existing = best.get(key)
        if existing is None or (
            _OWNERSHIP_RANK.get(tc.domain, len(_OWNERSHIP_RANK))
            < _OWNERSHIP_RANK.get(existing.domain, len(_OWNERSHIP_RANK))
        ):
            best[key] = tc
    return list(best.values())


def _parse_consolidated_json(raw: str) -> dict:
    """Extract JSON from the LLM's merger response."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in merger response")
    return json.loads(text[start:end])


def _fallback_merge(
    survivors: list[CandidatePlan],
    research_context: ResearchContext,
    run_id: str,
) -> ConsolidatedPlan:
    """Deterministic merge used when the LLM merger call fails."""
    # Union of all active domains across survivors
    domains: list[str] = sorted({
        domain
        for p in survivors
        for domain, active in p.domain_activations.items()
        if active
    })

    # Union of all tool calls across survivors, then collapse any tool+params
    # call that multiple survivors assigned to different domains down to one
    # (highest-priority domain wins — see _dedupe_cross_domain).
    planned_calls: list[PlannedToolCall] = [
        PlannedToolCall(
            tool=tc.get("tool", ""),
            params=tc.get("arguments", {}),
            domain=tc.get("domain", "general_search"),
        )
        for p in survivors
        for tc in p.tool_calls
    ]
    planned_calls = _dedupe_cross_domain(planned_calls)

    top = survivors[0]
    avg_feasibility = sum(p.feasibility_score for p in survivors) / len(survivors)
    avg_quality = sum(p.quality_score for p in survivors) / len(survivors)
    combined_gaps = " ".join(
        g for p in survivors for g in (p.gap_report if isinstance(p.gap_report, list) else [p.gap_report])
        if g
    )

    entity_manifest: dict = {
        "competitors": research_context.competitors,
        "operators": research_context.operators,
        "demand_side_companies": research_context.demand_side_companies,
        "tickers": research_context.tickers,
        "commodities": research_context.commodities,
        "mine_sites": research_context.mine_sites,
        "regions": research_context.regions,
        "news_queries": research_context.news_signals,
    }

    planned_calls = _expand_per_ticker_calls(planned_calls, entity_manifest)

    return ConsolidatedPlan(
        plan_id=f"consolidated-{run_id}",
        source_plan_ids=[p.plan_id for p in survivors],
        domains_active=domains,
        entity_manifest=entity_manifest,
        planned_tool_calls=planned_calls,
        research_findings=research_context.news_signals[0] if research_context.news_signals else "",
        rationale=top.rationale,
        gap_report=combined_gaps,
        feasibility_score=round(avg_feasibility, 3),
        quality_score=round(avg_quality, 3),
    )


class PlanMerger:
    """Synthesises TOT_SURVIVORS depth-2 plans into one ConsolidatedPlan."""

    def __init__(self) -> None:
        self._llm = LLMClient()
        self.last_usage: dict = {}

    async def merge(
        self,
        survivors: list[CandidatePlan],
        research_context: ResearchContext,
        run_id: str,
    ) -> ConsolidatedPlan:
        """Merge survivors into a single ConsolidatedPlan.

        Falls back to a deterministic union merge if the LLM call fails.
        """
        if not survivors:
            raise ValueError("PlanMerger.merge: no survivors to merge")

        # Sort highest combined_score first so the LLM prompt prioritises it
        sorted_survivors = sorted(survivors, key=lambda p: p.combined_score, reverse=True)

        research_findings_text = self._format_research(research_context)
        survivor_dicts = [
            {k: v for k, v in p.model_dump().items() if k != "gap_report"}
            | {"gap_report": " ".join(p.gap_report) if isinstance(p.gap_report, list) else p.gap_report}
            for p in sorted_survivors
        ]

        messages = plan_merge_messages(
            survivors=survivor_dicts,
            research_findings=research_findings_text,
            run_id=run_id,
        )

        try:
            response = await self._llm.acomplete(
                messages,
                temperature=settings.llm.work_temperature,
            )
            self.last_usage = response.usage
            data = _parse_consolidated_json(response.content or "")
            # Ensure plan_id is correct
            data["plan_id"] = f"consolidated-{run_id}"
            # Coerce planned_tool_calls
            raw_calls = data.pop("planned_tool_calls", [])
            planned_calls = [
                PlannedToolCall(
                    tool=tc.get("tool", ""),
                    params=tc.get("params", tc.get("arguments", {})),
                    domain=tc.get("domain", "general_search"),
                    rationale=tc.get("rationale", ""),
                )
                for tc in raw_calls
                if tc.get("tool")
            ]
            # Use research_context as the authoritative ticker source rather than
            # the LLM's own entity_manifest output, which may omit tickers it
            # didn't end up calling.
            planned_calls = _expand_per_ticker_calls(
                planned_calls, {"tickers": research_context.tickers},
            )
            # The LLM merger is asked to dedupe tool calls but isn't reliable about
            # catching the same tool+params assigned to two different domains under
            # different "angles" — enforce that deterministically.
            planned_calls = _dedupe_cross_domain(planned_calls)
            consolidated = ConsolidatedPlan(
                **{k: v for k, v in data.items() if k in ConsolidatedPlan.model_fields},
                planned_tool_calls=planned_calls,
            )
            # The LLM may omit demand-side consumers from its manifest; carry them
            # through from the authoritative research context so they aren't lost.
            if research_context.demand_side_companies:
                consolidated.entity_manifest.setdefault(
                    "demand_side_companies", research_context.demand_side_companies
                )
            return consolidated
        except Exception as exc:
            logger.warning("PlanMerger: LLM merge failed (%s) — using fallback deterministic merge", exc)
            return _fallback_merge(sorted_survivors, research_context, run_id)

    @staticmethod
    def _format_research(ctx: ResearchContext) -> str:
        if not any([ctx.competitors, ctx.operators, ctx.tickers, ctx.commodities, ctx.news_signals]):
            return "(no pre-planning research was conducted)"
        parts = []
        if ctx.competitors:
            parts.append(f"Competitors: {', '.join(ctx.competitors)}")
        if ctx.operators:
            parts.append(f"Mining operators: {', '.join(ctx.operators)}")
        if ctx.demand_side_companies:
            parts.append(
                "Demand-side consumers (route to macro_geopolitics): "
                + ", ".join(ctx.demand_side_companies)
            )
        if ctx.tickers:
            parts.append(f"Tickers: {', '.join(ctx.tickers)}")
        if ctx.commodities:
            parts.append(f"Commodities: {', '.join(ctx.commodities)}")
        if ctx.news_signals:
            parts.append("News: " + "; ".join(ctx.news_signals))
        if ctx.open_questions:
            parts.append("Open questions: " + "; ".join(ctx.open_questions))
        return "\n".join(parts)
