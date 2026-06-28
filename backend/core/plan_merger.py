"""Merges the top-N ToT survivor plans into a single ConsolidatedPlan for Gate 1.

Instead of asking the user to pick one of three plans, the merger synthesises
the survivors into one comprehensive plan that takes the best coverage from each.
"""
from __future__ import annotations

import json
import logging
import re

from config import settings
from core.domains import leaf_tools
from core.tot.schemas import (
    CandidatePlan,
    ConsolidatedPlan,
    DATASET_OWNERSHIP_PRIORITY,
    PlannedToolCall,
    ResearchContext,
    ResearchLeaf,
)
from models.llm_client import LLMClient
from prompts.plan_merge_prompt import plan_merge_messages
from services.masterdata_service import MasterDataService

logger = logging.getLogger(__name__)

# Each entity_manifest bucket maps to the domain + leaf type used when an entity
# is *not* found in master data (a research-surfaced rival, mine site, country, …).
# Master-data entities are resolved canonically instead (resolve_entity), which is
# what makes a company live under exactly one domain.
_MANIFEST_BUCKETS: list[tuple[str, str, str]] = [
    ("competitors", "competition", "company"),
    ("operators", "mining_operators", "company"),
    ("demand_side_companies", "general_search", "company"),
    ("commodities", "commodities", "commodity"),
    ("mine_sites", "mining_projects", "mine_site"),
    ("regions", "macroeconomics", "country"),
]

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

    Without this, e.g. both "commodities" and "macroeconomics" can independently
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


def _slug_key(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or label.lower()


def build_entity_manifest(research_context: ResearchContext) -> dict:
    """Flatten the research context into the entity manifest the planner and the
    leaf builder consume. Shared so a preliminary plan can be emitted mid-planning
    (understand_node) with the same shape the merger produces at the end."""
    return {
        "competitors": research_context.competitors,
        "operators": research_context.operators,
        "demand_side_companies": research_context.demand_side_companies,
        "tickers": research_context.tickers,
        "commodities": research_context.commodities,
        "mine_sites": research_context.mine_sites,
        "regions": research_context.regions,
        "news_queries": research_context.news_signals,
    }


def build_preliminary_leaves(research_context: ResearchContext) -> list[ResearchLeaf]:
    """Domain→leaf skeleton derivable from research alone, before tool calls are
    assigned. Lets the collection-plan tree build up during Understand."""
    return _build_leaves(build_entity_manifest(research_context), MasterDataService())


def _build_leaves(
    entity_manifest: dict, masterdata: MasterDataService
) -> list[ResearchLeaf]:
    """Turn the flat entity manifest into the structured stem→leaf plan.

    Every entity is resolved against master data first: a hit pins it to its
    canonical domain + leaf type (so Caterpillar can only ever be a `competition`
    company leaf), and a miss falls back to the manifest bucket it came from.
    Leaves are de-duplicated by (domain, key); each leaf's tools come from its
    leaf type (core/domains.LEAF_TOOLSETS), making collection an execution phase.
    """
    leaves: dict[tuple[str, str], ResearchLeaf] = {}
    order: list[tuple[str, str]] = []

    def add(label: str, domain: str, leaf_type: str, key: str, params: dict) -> None:
        dk = (domain, key)
        if dk in leaves:
            return
        leaves[dk] = ResearchLeaf(
            key=key, label=label, leaf_type=leaf_type, domain=domain,
            tools=leaf_tools(leaf_type), params=params,
        )
        order.append(dk)

    for bucket, fb_domain, fb_leaf_type in _MANIFEST_BUCKETS:
        for raw in entity_manifest.get(bucket) or []:
            label = str(raw).strip()
            if not label:
                continue
            res = masterdata.resolve_entity(label)
            if res is not None:
                add(res.label, res.domain, res.leaf_type, res.key, dict(res.params))
            else:
                add(label, fb_domain, fb_leaf_type, _slug_key(label), {})

    # Bare tickers only contribute a leaf when they resolve to a known entity;
    # unresolved tickers are still covered by per-ticker tool-call expansion.
    for t in entity_manifest.get("tickers") or []:
        tk = str(t).strip()
        if not tk:
            continue
        res = masterdata.resolve_entity(tk)
        if res is not None:
            add(res.label, res.domain, res.leaf_type, res.key, dict(res.params))

    return [leaves[k] for k in order]


def _recanonicalize_calls(
    calls: list[PlannedToolCall], masterdata: MasterDataService
) -> list[PlannedToolCall]:
    """Override each ticker-scoped call's domain with the entity's canonical domain.

    This is the execution-side half of the de-overlap fix: a `get_company_financials`
    call for CAT that the planner tagged "mining_operators" is moved to "competition"
    because Caterpillar resolves there in master data. Calls whose params don't name a
    resolvable entity (commodity symbols, FRED series, free-text queries) keep the
    planner's domain.
    """
    out: list[PlannedToolCall] = []
    for tc in calls:
        ticker = tc.params.get("ticker")
        if ticker:
            res = masterdata.resolve_entity(str(ticker))
            if res is not None and res.domain != tc.domain:
                tc = tc.model_copy(update={"domain": res.domain})
        out.append(tc)
    return out


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

    entity_manifest: dict = build_entity_manifest(research_context)

    planned_calls = _expand_per_ticker_calls(planned_calls, entity_manifest)

    masterdata = MasterDataService()
    planned_calls = _recanonicalize_calls(planned_calls, masterdata)
    planned_calls = _dedupe_cross_domain(planned_calls)
    leaves = _build_leaves(entity_manifest, masterdata)
    domains = sorted({lf.domain for lf in leaves} | set(domains))

    return ConsolidatedPlan(
        plan_id=f"consolidated-{run_id}",
        source_plan_ids=[p.plan_id for p in survivors],
        domains_active=domains,
        entity_manifest=entity_manifest,
        leaves=leaves,
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
            # Override each ticker-scoped call's domain with the entity's canonical
            # domain from master data, then collapse the same tool+params assigned
            # to two domains. Together these guarantee a company's calls land under
            # exactly one domain regardless of how the LLM tagged them.
            masterdata = MasterDataService()
            planned_calls = _recanonicalize_calls(planned_calls, masterdata)
            planned_calls = _dedupe_cross_domain(planned_calls)
            consolidated = ConsolidatedPlan(
                **{k: v for k, v in data.items()
                   if k in ConsolidatedPlan.model_fields and k != "leaves"},
                planned_tool_calls=planned_calls,
            )
            # The LLM may omit demand-side consumers from its manifest; carry them
            # through from the authoritative research context so they aren't lost.
            if research_context.demand_side_companies:
                consolidated.entity_manifest.setdefault(
                    "demand_side_companies", research_context.demand_side_companies
                )
            # Build the structured stem→leaf plan from the (now complete) manifest
            # and align domains_active with the domains the leaves actually cover.
            consolidated.leaves = _build_leaves(consolidated.entity_manifest, masterdata)
            consolidated.domains_active = sorted(
                {lf.domain for lf in consolidated.leaves} | set(consolidated.domains_active)
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
                "Demand-side consumers (route to general_search): "
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
