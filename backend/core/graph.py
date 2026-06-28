"""LangGraph state graph: Understand → Collect → Synthesize with ReAct backtracking.

Phase 6 changes:
- AgentState.plan (single dict) replaced by plans (list of survivor dicts).
- understand_node runs the full ToT pipeline:
    PlanProposer → GroundingAgent → score_and_prune → MCP state bus.
- collect_node runs all survivor plans in parallel via asyncio.gather.

Phase 7 changes:
- collect_node replaces the flat tool loop with per-domain CrewAI sub-agents.
  Each (plan, domain) pair is one parallel task; results are ChapterDrafts.
- synthesize_node calls core/merger.py to merge drafts, runs diversity recovery,
  then uses SynthesisAgent per domain for polished prose.
- AgentState gains chapter_sets and merge_log fields.

Phase 9 changes:
- AgentState gains cumulative_cost_usd, api_call_count, injection_flags,
  clarification_done fields.
- understand_node checks entity preferences before running ToT; pauses the run
  with stage="clarification_needed" if required entities are absent.
- collect_node enforces Guardrails (budget + call-count) before dispatching
  agents; routes to partial_brief_node on violation.
- synthesize_node scans retrieved chunks for prompt injection and filters them.
- partial_brief_node assembles whatever chapter_sets exist into a partial brief.
"""
import asyncio
import logging
import os
import time
from typing import TypedDict, Literal

import aiosqlite
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command  # noqa: F401 — Command re-exported for routers

from config import settings
from core.guardrails import Guardrails, ToolNotAllowed
from models.llm_client import LLMClient
from models.llm_retry import call_with_backoff
from models.usage import accumulate, llm_usage, merge_usage
from prompts.synthesize_prompt import exec_summary_messages
from core.event_logger import log_event, run_lock, update_run_context
from core import tool_circuit_breaker as breaker
from core.domains import domain_keys
from core.friendly_names import friendly_domain
from core.tot.proposer import PlanProposer
from core.tot.scorer import score_and_prune
from core.tot.schemas import ResearchContext
from core.research_agent import ResearchAgent
from core.plan_merger import PlanMerger, build_entity_manifest, build_preliminary_leaves
from agents.grounding_agent import GroundingAgent
from state_bus.server import clear_plans, write_plan_direct, get_all_plans_direct
from memory.context_window import ContextWindow
from memory.sqlite_store import SqliteStore
from core.merger import assign_global_citation_ids, merge_chapter_sets, chapter_set_overlap
from core.schemas import ChapterDraft, MergedChapter
from core.subdomains import (
    enumerate_subdomains,
    assemble_entity_evidence,
    group_datasets_by_entity,
    classify_entity_domain,
)
from tools.registry import tool_display_name
from core import completeness
from core.tool_router import async_route
from services.masterdata_service import MasterDataService
from agents.synthesis_agent import SynthesisAgent
from retrieval import Retriever
from retrieval.chunker import Chunker

logger = logging.getLogger(__name__)

_DOMAINS = domain_keys()

_UNSTRUCTURED_TOOLS = {
    "news_search",
    "web_search",
    "web_extract",
    "web_crawl",
    "web_map",
    "web_research",
    "sec_filings",
}

# Required preference keys for entity clarification
_REQUIRED_ENTITY_PREFS = ("equipment_models", "operators", "competitor_tickers")


class AgentState(TypedDict):
    run_id: str
    session_id: str
    user_query: str
    plans: list                          # list of survivor CandidatePlan dicts
    collection_manifest: dict            # {domain: [result_dict, ...]} (backward-compat)
    chapter_sets: dict                   # {"plan_id::domain": ChapterDraft.model_dump()}
    synthesis_chapters: list             # [{"domain": str, "text": str}, ...]
    merge_log: list[str]                 # contradiction resolution log from merger
    confidence: float
    react_iterations: int
    context_messages: list               # [{role, content, token_count}]
    stage: Literal[
        "understand", "collect", "synthesize", "done", "error",
        "partial", "clarification_needed",
    ]
    warnings: list[str]
    error: str | None
    # Phase 9 safety fields
    cumulative_cost_usd: float
    api_call_count: int
    # Real token accounting for the live cost counter
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    injection_flags: list[str]
    clarification_done: bool
    # Phase 10 report fields
    exec_summary: str
    # {id: citation_dict} — every unique source across all domains, numbered once
    # by assign_global_citation_ids() so the same source shares one id everywhere.
    citation_registry: dict
    # Soft-timeout tracking: unix timestamp set once at graph start
    run_start_time: float
    # Cumulative seconds spent paused at gates/timeout-checks, excluded from elapsed-time math
    paused_seconds: float
    # Number of times the user has clicked "continue" on a timeout prompt;
    # the next prompt is due at (timeout_prompt_count + 1) * soft_timeout_s of active elapsed time
    timeout_prompt_count: int
    # Set by the stall-watchdog "finalize" action: when True, the long-running nodes
    # short-circuit to the partial-brief path instead of doing more work, so a stalled
    # run can be pushed to a report + the next gate instead of hanging forever.
    force_finalize: bool
    # Free-text guidance the user typed when rejecting the Gate-1 plan ("you forgot
    # Liebherr / add tariff data"); folded into the next planning pass, then cleared.
    redirect_feedback: str | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _active_domains(plan: dict) -> list[str]:
    """Return the list of domains marked active in a plan.

    Supports ConsolidatedPlan (`domains_active: list[str]`) and the legacy
    CandidatePlan (`domain_activations: dict[str, bool]`) shapes.
    """
    if plan.get("domains_active"):
        return [d for d in plan["domains_active"] if d in _DOMAINS]
    return [
        d for d in _DOMAINS
        if plan.get("domain_activations", {}).get(d, False)
    ]


def _draft_ok(draft: dict | None) -> bool:
    """A draft is reusable if it has real content and no tool errors.

    Used by collect_node to avoid re-running domains that already succeeded when the
    graph backtracks for a low-confidence retry.
    """
    if not draft:
        return False
    text = (draft.get("text") or "").strip()
    if not text or text.startswith(("No tool calls assigned", "No data collected")):
        return False
    return not draft.get("tool_errors")


# Row-level data isn't useful on a human "does this look right" review screen and
# blows up the Gate 2 payload (e.g. a 500-row OHLCV table per company); keep a
# small preview plus the count/columns so the reviewer still sees shape and a
# sample, without embedding the full series for every entity.
_GATE2_ROW_PREVIEW = 5
_GATE2_SUMMARY_PREVIEW_CHARS = 1200
_GATE2_SUMMARY_TRUNCATION_NOTE = (
    "\n\n[Preview truncated for Gate 2; full extracted content remains available "
    "to synthesis.]"
)


def _gate2_dataset_view(dataset: dict) -> dict:
    """Return a compact Gate-2 preview of a dataset without mutating graph state."""
    trimmed: dict | None = None

    rows = dataset.get("rows")
    if isinstance(rows, list) and len(rows) > _GATE2_ROW_PREVIEW:
        trimmed = dict(dataset)
        trimmed["rows"] = rows[:_GATE2_ROW_PREVIEW]
        trimmed["rows_truncated"] = True

    summary = dataset.get("summary")
    if isinstance(summary, str) and len(summary) > _GATE2_SUMMARY_PREVIEW_CHARS:
        if trimmed is None:
            trimmed = dict(dataset)
        trimmed["summary"] = (
            summary[:_GATE2_SUMMARY_PREVIEW_CHARS].rstrip()
            + _GATE2_SUMMARY_TRUNCATION_NOTE
        )
        trimmed["summary_truncated"] = True

    if trimmed is None:
        return dataset
    return trimmed


def _draft_source_entries(
    domain: str,
    draft: dict,
    masterdata=None,
    demand_side_companies: list[str] | None = None,
) -> list[dict]:
    """Turn one domain draft into enriched Sources-panel entries.

    One entry per collected dataset (typed: data_type + label + count) and one per
    failed tool (rendered in red on the frontend), so the live panel reflects *what*
    was collected and *what was attempted but failed* — not just a list of URLs.

    Each dataset is cross-checked against every master-data-backed domain so a
    dataset collected by one domain's agent but actually about an entity belonging
    to another domain (e.g. a Caterpillar mention surfaced while collecting
    "mining_operators") is filed under the right domain instead.
    """
    entries: list[dict] = []
    for ds in draft.get("datasets") or []:
        true_domain, segment = (
            classify_entity_domain(ds, masterdata, demand_side_companies)
            if masterdata is not None else (None, "")
        )
        entries.append({
            "domain": true_domain or domain,
            "tool": tool_display_name(ds.get("tool", "")),
            "title": ds.get("title") or "",
            "data_type": ds.get("data_type", "data"),
            "label": ds.get("label", ""),
            "count": ds.get("count", 0),
            "url": None,
            "published_at": None,
            "failed": False,
            "segment": segment,
        })
    for ft in draft.get("failed_tools") or []:
        display = ft.get("tool_display") or tool_display_name(ft.get("tool", ""))
        entries.append({
            "domain": domain,
            "tool": display,
            "title": display,
            "data_type": "failed",
            "label": display,
            "count": 0,
            "url": None,
            "published_at": None,
            "failed": True,
            "reason": ft.get("reason", ""),
        })
    # A domain that only produced cited links (no normalized dataset) still appears.
    if not entries and (draft.get("citations")):
        n = len(draft["citations"])
        entries.append({
            "domain": domain,
            "tool": tool_display_name("web_search"),
            "title": f"{n} link(s)",
            "data_type": "web_results",
            "label": "",
            "count": n,
            "url": None,
            "published_at": None,
            "failed": False,
        })
    return entries


async def _persist_partial_sources(
    run_id: str, session_id: str, query: str, domain: str, draft: "ChapterDraft",
    masterdata=None, demand_side_companies: list[str] | None = None,
) -> None:
    """Append one domain's sources to the run row as soon as its agent finishes,
    so the frontend Sources panel can populate live during collection instead of
    waiting for the whole collect_node (and Gate 2) to complete."""
    new_entries = _draft_source_entries(domain, draft.model_dump(), masterdata, demand_side_companies)
    if not new_entries:
        return
    store = SqliteStore()
    async with run_lock(run_id):
        row = await store.get_run(run_id)
        existing = list(row.get("sources") or []) if row else []
        # Drop this domain's provisional per-tool rows (written live by the tool
        # router) so they're replaced by the typed datasets, not duplicated.
        existing = [
            s for s in existing
            if not (s.get("provisional") and s.get("domain") == domain)
        ]
        existing.extend(new_entries)
        await store.upsert_run(
            run_id, session_id, query,
            status="running", stage="collect",
            sources=existing,
        )


async def _run_domain_agent(
    plan: dict,
    domain: str,
    run_id: str,
    retriever: Retriever,
    chunker: Chunker,
    collection: str,
) -> tuple[str, str, ChapterDraft]:
    """Run one domain sub-agent for one survivor plan.

    Returns (plan_id, domain, ChapterDraft).
    Writes unstructured chapter text to ChromaDB (kept in core/, not in agents/).
    """
    # Lazy import avoids circular import at module load time
    from agents import DOMAIN_AGENTS

    plan_id = plan.get("plan_id", "unknown")
    agent_cls = DOMAIN_AGENTS.get(domain)
    if agent_cls is None:
        logger.warning("_run_domain_agent: no agent registered for domain '%s'", domain)
        draft = ChapterDraft(
            domain=domain,
            plan_id=plan_id,
            text=f"No agent registered for domain {domain}.",
        )
        return plan_id, domain, draft

    agent_instance = agent_cls()
    draft = await agent_instance.run(plan, run_id)

    # Write chapter text to the per-run Chroma collection (unstructured store) so
    # later synthesis passes can retrieve it as additional context. This is the
    # system's own prior synthesis being re-embedded, not an external source, so
    # it's tagged with an internal marker that synthesis_agent._format_chunks
    # recognizes and never lets a citation-hungry LLM mistake for a real source.
    if draft.text.strip():
        try:
            docs = chunker.chunk_text(draft.text, source="__internal_synthesis__", domain=domain)
            retriever.add(collection, docs)
        except Exception as exc:
            logger.warning("_run_domain_agent: chroma write failed for %s/%s: %s",
                           domain, plan_id, exc)

    return plan_id, domain, draft


# ---------------------------------------------------------------------------
# Soft-timeout helpers
# ---------------------------------------------------------------------------

class _TimeoutStopSignal(Exception):
    """Raised inside a node to request a graceful stop at the current boundary."""


def _timeout_interrupt_if_needed(state: AgentState) -> None:
    """Call at the very start of every node.

    If the run has exceeded the soft timeout, suspends the graph via
    interrupt() so the frontend can ask the user whether to continue.
    On 'approve' the call returns normally and the node proceeds.
    On 'redirect' raises _TimeoutStopSignal so the caller can return
    {"stage": "partial"} and route toward partial_brief or END.
    """
    start = state.get("run_start_time")
    if not start:
        return
    elapsed = int(time.time() - start - state.get("paused_seconds", 0.0))
    next_threshold = settings.react.soft_timeout_s * (state.get("timeout_prompt_count", 0) + 1)
    if elapsed <= next_threshold:
        return
    minutes = elapsed // 60
    decision = interrupt({
        "type": "timeout_check",
        "elapsed_s": elapsed,
        "message": (
            f"The agent has been running for {minutes} minute(s). "
            "Continue research, or stop and receive a partial report with findings so far?"
        ),
    })
    if decision == "redirect":
        raise _TimeoutStopSignal()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

async def understand_node(state: AgentState) -> dict:
    if state.get("force_finalize"):
        return {"stage": "partial"}
    try:
        _timeout_interrupt_if_needed(state)
    except _TimeoutStopSignal:
        return {"stage": "partial"}

    query = state["user_query"]

    # Fold any Gate-1 rejection feedback ("you forgot Liebherr / add tariff data")
    # into the planning input for this pass, so the replan actually covers it.
    feedback = (state.get("redirect_feedback") or "").strip()
    if feedback:
        query = f"{query}\n\nAdditional user guidance: {feedback}"
        await log_event("progress", "Replanning with your additional guidance…")

    # --- Phase 9.3: Entity clarification gate (optional, off by default) ---
    if settings.gates.gate_clarification_enabled and not state.get("clarification_done"):
        store = SqliteStore()
        missing = []
        for key in _REQUIRED_ENTITY_PREFS:
            if not await store.get_preference(key):
                missing.append(key)
        if missing:
            logger.info("understand_node: missing entity preferences %s — pausing for clarification", missing)
            return {
                "stage": "clarification_needed",
                "error": f"Missing required entity preferences: {missing}",
            }

    retriever = Retriever()

    # 1. Retrieve background context from the industry knowledge base
    try:
        rag_chunks, _ = retriever.retrieve(
            query,
            settings.stores.chroma_knowledge_collection,
            top_k=5,
        )
    except Exception:
        rag_chunks = []

    # 2. Pre-planning research loop: resolve entities + discover market context
    research_context = ResearchContext()
    researcher_usage: dict = {}
    if settings.understand.research_enabled:
        try:
            researcher = ResearchAgent()
            research_context = await researcher.run(query)
            researcher_usage = researcher.last_usage
            logger.info(
                "understand_node: research found %d competitors, %d operators, "
                "%d tickers, %d commodities",
                len(research_context.competitors),
                len(research_context.operators),
                len(research_context.tickers),
                len(research_context.commodities),
            )
            await log_event(
                "progress",
                f"Found {len(research_context.competitors)} competitors, "
                f"{len(research_context.operators)} mining operators, and "
                f"{len(research_context.commodities)} commodities to investigate.",
            )
        except Exception as exc:
            logger.warning("understand_node: ResearchAgent failed (%s) — proceeding without research", exc)

    # Stream the plan skeleton to the UI the moment research resolves entities:
    # persist a preliminary plan (domains + leaves, no tool calls yet) so the
    # collection-plan tree builds up during Understand instead of materialising all
    # at once at Gate 1. The full consolidated plan (with tool calls) overwrites this
    # after the merge below. Best-effort — never block planning on it.
    try:
        prelim_leaves = build_preliminary_leaves(research_context)
        if prelim_leaves:
            prelim_plan = {
                "plan_id": f"consolidated-{state['run_id']}",
                "domains_active": sorted({lf.domain for lf in prelim_leaves}),
                "entity_manifest": build_entity_manifest(research_context),
                "leaves": [lf.model_dump() for lf in prelim_leaves],
                "planned_tool_calls": [],
            }
            await SqliteStore().upsert_run(
                state["run_id"], state.get("session_id", ""), state["user_query"],
                status="running", stage="understand", plans=[prelim_plan],
            )
    except Exception as exc:
        logger.debug("understand_node: preliminary plan persist failed (non-fatal): %s", exc)

    # 3. Propose 7 depth-1 plans (enriched with research_context)
    await log_event("progress", "Drafting candidate research plans…")
    try:
        proposer = PlanProposer()
        candidates = await proposer.propose(query, rag_chunks, research_context)
    except Exception as exc:
        logger.exception("understand_node: PlanProposer failed")
        await log_event(
            "error", "PlanProposer failed",
            detail={"error": str(exc), "exc_type": type(exc).__name__},
            level="error",
        )
        return {"stage": "error", "error": f"PlanProposer failed: {exc}"}

    # 4-7. Ground to depth-2, score/prune to survivors, merge into one
    # consolidated plan. If the consolidated plan still reports gaps
    # (GroundingAgent couldn't find a substitute for an unavailable tool),
    # retry grounding+merge up to gap_remediation_max_rounds extra times
    # before handing the plan to Gate 1. This is bounded by round count, not
    # time — most gaps are structural and won't close with repeated
    # attempts — and reuses the same soft-timeout check-in as every other
    # node, so a long remediation pass still gets the usual 15-min prompt.
    grounder_usage: dict = {}
    merger_usage: dict = {}
    max_rounds = settings.understand.gap_remediation_max_rounds
    remediation_round = 0

    while True:
        await log_event("progress", "Reviewing and refining the research plans…")
        try:
            grounder = GroundingAgent()
            grounded = grounder.run(candidates)
            grounder_usage = merge_usage(grounder_usage, grounder.last_usage)
        except Exception as exc:
            logger.warning(
                "understand_node: GroundingAgent failed (%s) — skipping grounding.", exc
            )
            grounded = candidates  # use depth-1 plans as fallback

        # 5. Score and prune to survivors
        all_plans = score_and_prune(grounded)
        survivors = [p for p in all_plans if p.is_survivor]

        # 6. Write all plans to the MCP state bus; clear stale plans first
        clear_plans()
        for plan in all_plans:
            write_plan_direct(plan)

        # 7. Merge survivors into one consolidated plan
        await log_event("progress", "Selecting the strongest research plan…")
        try:
            merger = PlanMerger()
            consolidated = await merger.merge(survivors, research_context, run_id=state["run_id"])
            merger_usage = merge_usage(merger_usage, merger.last_usage)
            logger.info(
                "understand_node: merged %d survivors into consolidated plan '%s'",
                len(survivors),
                consolidated.plan_id,
            )
        except Exception as exc:
            logger.warning("understand_node: PlanMerger failed (%s) — falling back to top survivor only", exc)
            # Fallback: use the top survivor as-is, converted to a ConsolidatedPlan-like dict
            top = survivors[0] if survivors else None
            if top is None:
                await log_event(
                    "error", "No survivor plans and merger failed",
                    detail={"error": str(exc), "exc_type": type(exc).__name__},
                    level="error",
                )
                return {"stage": "error", "error": "No survivor plans and merger failed"}
            from core.tot.schemas import ConsolidatedPlan
            consolidated = ConsolidatedPlan(
                plan_id=f"consolidated-{state['run_id']}",
                source_plan_ids=[top.plan_id],
                domains_active=[d for d, a in top.domain_activations.items() if a],
                entity_manifest={},
                planned_tool_calls=[],
                research_findings="",
                rationale=top.rationale,
                gap_report=" ".join(top.gap_report) if isinstance(top.gap_report, list) else top.gap_report,
                feasibility_score=top.feasibility_score,
                quality_score=top.quality_score,
            )
            break  # no merger available to retry against

        gaps = consolidated.gap_report
        has_gaps = bool(gaps.strip()) if isinstance(gaps, str) else bool(gaps)
        if not has_gaps or remediation_round >= max_rounds:
            break

        remediation_round += 1
        logger.info(
            "understand_node: consolidated plan has unresolved gaps, retrying "
            "grounding (remediation round %d/%d)",
            remediation_round, max_rounds,
        )
        try:
            _timeout_interrupt_if_needed(state)
        except _TimeoutStopSignal:
            return {"stage": "partial"}

    n_candidates = len(all_plans)
    n_survivors = len(survivors)
    cw = ContextWindow.from_state(
        state.get("context_messages", []), settings.react.run_token_budget
    )
    cw.add(
        "assistant",
        f"ToT: {n_candidates} plans proposed, {n_survivors} survivors merged into {consolidated.plan_id}.",
    )

    usage_delta = accumulate(state, proposer.last_usage, grounder_usage, researcher_usage, merger_usage)

    store = SqliteStore()
    auto_approve = settings.gates.auto_approve_gates or bool(await store.get_preference("auto_approve_gates"))
    gate1_needed = settings.gates.gate_1_enabled and not auto_approve

    # Pass the consolidated plan forward. collect_node iterates `plans` as a list;
    # wrapping in a list of one preserves backward compatibility. Gate 1's
    # interrupt() lives in its own node (gate1_node) rather than here, since
    # LangGraph replays a node from the top on resume — interrupting after all
    # this expensive ToT work would silently redo it on every approval.
    return {
        "plans": [consolidated.model_dump()],
        "stage": "understand" if gate1_needed else "collect",
        "context_messages": cw.to_state(),
        # Consumed above; clear it so it can't bleed into a later replan.
        "redirect_feedback": None,
        **usage_delta,
    }


async def gate1_node(state: AgentState) -> dict:
    """Gate 1 — human review of the consolidated research plan.

    Split out of understand_node so that approving the gate doesn't replay the
    expensive ToT planning pipeline (LangGraph re-runs a node from the top on
    resume). This node only re-reads the already-computed plan from state.
    """
    plans = state.get("plans") or []
    plan = plans[0] if plans else {}
    decision = interrupt({"gate": 1, "plan": plan})
    if decision == "redirect":
        return {"plans": [], "stage": "understand"}
    return {"stage": "collect"}


async def collect_node(state: AgentState) -> dict:
    # Tag progress events from this node with the collect stage (see synthesize_node).
    update_run_context(stage="collect")
    if state.get("force_finalize"):
        # Stall watchdog → finalize: stop collecting and route to the partial brief,
        # which synthesizes whatever drafts were already committed by a prior pass.
        return {"stage": "partial"}
    try:
        _timeout_interrupt_if_needed(state)
    except _TimeoutStopSignal:
        return {"stage": "partial"}

    plans = state.get("plans") or []
    if not plans:
        await log_event(
            "error", "collect_node: no plans in state",
            level="error",
        )
        return {"stage": "error", "error": "collect_node reached with no plans"}

    retriever = Retriever()
    chunker = Chunker(settings.retrieval.chunk_size, settings.retrieval.chunk_overlap)
    collection = f"{settings.stores.chroma_collected_prefix}_{state['run_id']}"

    # Used to re-check each collected dataset against competitor/customer/third-party
    # masterdata so it lands under the domain its content actually belongs to, not
    # just the domain of the agent that happened to collect it.
    masterdata = MasterDataService()
    primary_entity_manifest = plans[0].get("entity_manifest", {}) if plans else {}
    demand_side_companies = primary_entity_manifest.get("demand_side_companies") or []

    sem = asyncio.Semaphore(settings.max_parallel_subagents)

    async def bounded(plan: dict, domain: str) -> tuple[str, str, ChapterDraft]:
        async with sem:
            return await _run_domain_agent(plan, domain, state["run_id"],
                                           retriever, chunker, collection)

    # --- Tool allowlist enforcement (no cost/call caps — tracking only) ---
    guardrails = Guardrails()

    # Carry over drafts from a previous collect pass so a backtrack only re-runs
    # domains that have not yet produced a good draft (smarter backtrack).
    prior_sets: dict[str, dict] = dict(state.get("chapter_sets") or {})

    # Build the full active+allowed (plan, domain) pair list. This drives both the real
    # "no active domains" error and the confidence denominator.
    skipped_domains: list[str] = []
    pairs: list[tuple[dict, str, str]] = []
    for plan in plans:
        plan_id = plan.get("plan_id", "unknown")
        for domain in _active_domains(plan):
            try:
                guardrails.check_tool_allowed(domain)
            except ToolNotAllowed:
                skipped_domains.append(domain)
                logger.info("collect_node: domain '%s' blocked by tool allowlist", domain)
                continue
            pairs.append((plan, domain, f"{plan_id}::{domain}"))

    if skipped_domains:
        logger.warning("collect_node: skipped disallowed domains: %s", skipped_domains)

    if not pairs:
        await log_event(
            "error", "collect_node: no active domains in any survivor plan",
            detail={"skipped_domains": skipped_domains},
            level="error",
        )
        return {"stage": "error", "error": "No active domains in any survivor plan"}

    # Reuse good prior drafts; only schedule domains that still need (re)collection.
    chapter_sets: dict[str, dict] = {}
    tasks = []
    reused = 0
    for plan, domain, key in pairs:
        if _draft_ok(prior_sets.get(key)):
            chapter_sets[key] = prior_sets[key]
            reused += 1
        else:
            tasks.append(bounded(plan, domain))

    if reused:
        logger.info("collect_node: reusing %d prior successful draft(s); re-running %d",
                    reused, len(tasks))

    draft_usages: list[dict] = []
    collected_tool_errors: list[str] = []

    for fut in asyncio.as_completed(tasks):
        try:
            plan_id, domain, draft = await fut
        except Exception as exc:
            logger.warning("collect_node: domain agent raised %s", exc)
            collected_tool_errors.append(f"domain agent crashed: {exc}")
            continue
        chapter_sets[f"{plan_id}::{domain}"] = draft.model_dump()
        collected_tool_errors.extend(draft.tool_errors)
        draft_usages.append(draft.usage)
        await _persist_partial_sources(
            state["run_id"], state["session_id"], state["user_query"], domain, draft,
            masterdata, demand_side_companies,
        )

    # Build the sources manifest from the full merged chapter_sets (carried + new), so the
    # sources panel reflects everything collected across iterations. Entries are already
    # in the enriched Sources-panel shape (typed datasets + failed tools); the API layer
    # just flattens them onto the run row.
    manifest: dict[str, list] = {d: [] for d in _DOMAINS}
    for key, draft in chapter_sets.items():
        _plan_id, _, domain = key.partition("::")
        manifest.setdefault(domain, []).extend(
            _draft_source_entries(domain, draft, masterdata, demand_side_companies)
        )

    if collected_tool_errors:
        logger.warning("collect_node: %d tool/agent failures: %s",
                       len(collected_tool_errors), collected_tool_errors)

    # Tool health: surface which tools failed (and which were blocked by the circuit
    # breaker, and why) so the failure rate is visible rather than buried in retries.
    health = breaker.summary(state["run_id"])
    if health:
        blocked = [t for t, h in health.items() if h["blocked"]]
        await log_event(
            "tool_health",
            (f"Tool health: {len(blocked)} tool(s) blocked this run"
             if blocked else "Tool health: failures recorded but none blocked"),
            detail={"tools": health, "blocked": blocked},
            level="warning" if blocked else "info",
        )

    # Confidence over the union of all attempted pairs (carried + newly run), so it rises
    # across backtracks and routes to synthesis once every domain has a good draft.
    total = len(pairs)
    successful = sum(1 for _, _, key in pairs if _draft_ok(chapter_sets.get(key)))
    confidence = successful / total if total > 0 else 0.0
    usage_delta = accumulate(state, *draft_usages)

    cw = ContextWindow.from_state(
        state.get("context_messages", []), settings.react.run_token_budget
    )
    cw.add(
        "assistant",
        f"Phase 7 collect: {successful}/{total} domain agents succeeded. "
        f"Confidence: {confidence:.2f}.",
    )

    return {
        "collection_manifest": manifest,
        "chapter_sets": chapter_sets,
        "confidence": confidence,
        "stage": "collect",
        "context_messages": cw.to_state(),
        "warnings": list(state.get("warnings", [])) + collected_tool_errors,
        **usage_delta,
    }


async def data_review_node(state: AgentState) -> dict:
    """Gate 2 — human review of the data gathered. Reached only once the silent
    collect/backtrack retry loop has settled (good confidence or retries exhausted),
    so the human is never re-prompted on transient tool failures.

    Surfaces the actual datasets collected per domain, grouped by the entity each
    one is about (e.g. Competition → Caterpillar / John Deere), plus any tool
    failures shown as gaps. On "redirect" the collection is reset and re-run; on
    approve the graph proceeds to synthesis.
    """
    store = SqliteStore()
    auto_approve = settings.gates.auto_approve_gates or bool(await store.get_preference("auto_approve_gates"))
    if not settings.gates.gate_2_enabled or auto_approve:
        return {"stage": "synthesize"}

    chapter_sets: dict[str, dict] = dict(state.get("chapter_sets") or {})
    plans = state.get("plans") or []
    primary_plan = plans[0] if plans else {}
    masterdata = MasterDataService()
    entity_manifest = primary_plan.get("entity_manifest", {}) if isinstance(primary_plan, dict) else {}
    demand_side_companies = entity_manifest.get("demand_side_companies") or []

    # Collect every domain's datasets/failed tools first, then re-check each dataset
    # against cross-domain masterdata so one that's actually about a competitor or
    # third-party demand company (e.g. surfaced by a customer agent's web search)
    # moves to the domain it actually belongs to before bucketing by entity.
    datasets_by_domain: dict[str, list[dict]] = {}
    failed_tools_by_domain: dict[str, list[dict]] = {}
    for domain in _DOMAINS:
        drafts = [v for k, v in chapter_sets.items() if k.endswith(f"::{domain}")]
        if not drafts:
            continue
        for d in drafts:
            failed_tools_by_domain.setdefault(domain, []).extend(d.get("failed_tools") or [])
            for ds in d.get("datasets") or []:
                true_domain, _segment = classify_entity_domain(
                    ds, masterdata, demand_side_companies, entity_manifest
                )
                datasets_by_domain.setdefault(true_domain or domain, []).append(
                    _gate2_dataset_view(ds)
                )

    domains_payload = []
    for domain in _DOMAINS:
        datasets = datasets_by_domain.get(domain, [])
        failed_tools = failed_tools_by_domain.get(domain, [])
        if not datasets and not failed_tools:
            continue
        entities = group_datasets_by_entity(domain, datasets, primary_plan, masterdata)
        domains_payload.append({
            "domain": domain,
            "entities": entities,
            "failed_tools": failed_tools,
        })

    decision = interrupt({"gate": 2, "domains": domains_payload})
    if decision == "redirect":
        return {"chapter_sets": {}, "confidence": 0.0, "react_iterations": 0, "stage": "collect"}
    return {"stage": "synthesize"}


async def backtrack_node(state: AgentState) -> dict:
    return {"react_iterations": state["react_iterations"] + 1}


async def partial_brief_node(state: AgentState) -> dict:
    """Assemble a partial brief from whatever chapter_sets exist when a guardrail fires."""
    all_warnings = list(state.get("warnings", []))

    # If synthesis already produced chapters (e.g. a soft timeout landed *after*
    # the synthesize node finished its work), keep them rather than rebuilding a
    # cruder brief from chapter_sets — and carry the executive summary along.
    existing = state.get("synthesis_chapters") or []
    if existing:
        return {
            "synthesis_chapters": existing,
            "stage": "done",
            "exec_summary": state.get("exec_summary", ""),
            "warnings": all_warnings,
            **accumulate(state),
        }

    chapter_sets = state.get("chapter_sets") or {}

    chapters: list[dict] = []
    synth_usages: list[dict] = []
    if chapter_sets:
        synth_agent = SynthesisAgent()
        sem = asyncio.Semaphore(settings.synthesis.max_parallel_chapters)

        async def _synth_partial(key: str, draft_dict: dict) -> dict:
            domain = draft_dict.get("domain") or (key.split("::")[-1] if "::" in key else key)
            mc = MergedChapter(domain=domain, text=draft_dict.get("text", ""))
            async with sem:
                try:
                    return await asyncio.to_thread(synth_agent.run, domain, mc, [], [])
                except Exception as exc:
                    logger.warning("partial_brief_node: synthesis failed for %s: %s", domain, exc)
                    return {"domain": domain, "text": mc.text}

        results = await asyncio.gather(
            *[_synth_partial(k, d) for k, d in chapter_sets.items()]
        )
        for result in results:
            synth_usages.append(result.get("usage", {}))
            chapters.append(result)
    else:
        all_warnings.append("Partial brief: no data was collected before the guardrail fired.")

    return {
        "synthesis_chapters": chapters,
        "stage": "done",
        "warnings": all_warnings,
        **accumulate(state, *synth_usages),
    }


def _dedupe_citations(citations: list[dict]) -> list[dict]:
    """Dedupe citation dicts by url (else title+publisher), preserving order."""
    seen: set = set()
    out: list[dict] = []
    for c in citations:
        url = (c.get("url") or "").strip().lower()
        key = ("url", url) if url else (
            "title", (c.get("title") or "").strip().lower(), (c.get("publisher") or "").strip().lower()
        )
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


async def _remediate_subchapters(
    mc: MergedChapter,
    subdomains: list,
    subchapters: list[dict],
    synth_agent: SynthesisAgent,
    query: str,
    retriever,
    collection: str,
    guardrails,
    all_warnings: list[str],
) -> list[dict]:
    """Completeness gate: bounded remediation for Tier-1 subchapters that fell
    back to a generic placeholder.

    Two cases, both handled here: (1) evidence existed but the CrewAI call
    failed (likely a transient rate-limit, now mitigated by the retry/backoff
    in synthesis_agent.py, so a single resynthesis attempt usually fixes it),
    and (2) there was no evidence at all, in which case one extra broadened
    web_search is tried before resynthesizing. Either way this runs at most
    `settings.synthesis.completeness_max_remediation_rounds` times per
    subchapter, then replaces any still-broken text with an honest message.
    """
    by_key = {s.key: s for s in subdomains}
    rounds = max(1, settings.synthesis.completeness_max_remediation_rounds)
    fixed: list[dict] = []

    for sc in subchapters:
        if not completeness.is_gap(sc):
            fixed.append(sc)
            continue

        label = sc.get("subdomain_label", sc.get("subdomain_key", ""))
        sub = by_key.get(sc.get("subdomain_key", ""))
        had_evidence = completeness.has_evidence(sc)

        for _ in range(rounds):
            figures = sc.get("figures") or {}
            datasets = sc.get("datasets") or []
            citations = sc.get("citations") or []
            chunks: list = []

            if sub is not None:
                evidence = assemble_entity_evidence(
                    sub, mc, retriever, collection, guardrails, query
                )
                figures = {**evidence.figures, **figures}
                datasets = datasets or evidence.datasets
                citations = _dedupe_citations(citations + evidence.citations)
                chunks = evidence.retrieved_chunks

                if not had_evidence:
                    try:
                        result = await async_route(
                            "web_search",
                            {"query": completeness.broadened_query(label, query), "max_results": 5},
                        )
                        items = [
                            {"title": r.get("title", ""), "url": r.get("url"),
                             "snippet": r.get("content", "")}
                            for r in (result.get("results") or [])[:5]
                        ]
                        if items:
                            datasets = datasets + [{
                                "tool": "web_search", "title": f"{len(items)} result(s)",
                                "kind": "list", "items": items,
                            }]
                            new_citations = [
                                {"id": None, "title": i.get("title") or i["url"],
                                 "url": i.get("url"), "publisher": None}
                                for i in items if i.get("url")
                            ]
                            citations = _dedupe_citations(citations + new_citations)
                    except Exception as exc:
                        all_warnings.append(
                            f"completeness_gate: {label} remediation search failed: {exc}"
                        )

            sc = await asyncio.to_thread(
                synth_agent.run_subchapter,
                mc.domain, sc.get("subdomain_key", ""), label,
                figures, datasets, citations, chunks, query,
            )
            if not completeness.is_gap(sc):
                all_warnings.append(f"completeness_gate: resynthesized {label} after initial gap")
                break
        else:
            all_warnings.append(f"completeness_gate: {label} still incomplete after remediation")
            err = sc.get("synthesis_error")
            sc["text"] = completeness.honest_fallback_message(label, [err] if err else None)

        fixed.append(sc)

    return fixed


async def synthesize_node(state: AgentState) -> dict:
    # Tag every progress event emitted from this node with the real stage; the
    # context var is otherwise left at "understand" (set once at run start), so
    # chapter-writing events would be mis-filed under the wrong phase.
    update_run_context(stage="synthesize")
    if state.get("force_finalize"):
        # Stall watchdog → finalize: route to the partial brief, which keeps any
        # synthesis_chapters already in state or rebuilds from the committed chapter_sets.
        return {"stage": "partial"}
    # Deliberately no soft-timeout interrupt here. Synthesis is the final
    # productive phase: pausing it to ask "continue or stop?" risks discarding
    # work that lives in local variables until this node's final return, and a
    # second interrupt in the same node (alongside Gate 3 below) lets a resume
    # value meant for one interrupt be consumed by the other. The soft timeout
    # is still enforced at the understand / collect / backtrack node starts.

    plans = state.get("plans") or []
    if not plans:
        return {"stage": "error", "error": "synthesize_node reached with no plans"}

    chapter_sets = state.get("chapter_sets") or {}
    query = state["user_query"]
    all_warnings: list[str] = list(state.get("warnings", []))
    all_injection_flags: list[str] = list(state.get("injection_flags", []))
    usage_dicts: list[dict] = []   # token usage from synthesis + recovery + exec summary

    retriever = Retriever()
    collection = f"{settings.stores.chroma_collected_prefix}_{state['run_id']}"
    _guardrails = Guardrails()

    # ------------------------------------------------------------------
    # Step 1: Merge chapter drafts from all survivor plans
    # ------------------------------------------------------------------
    merged_chapters: list[MergedChapter]
    merge_log: list[str]

    if chapter_sets:
        merged_chapters, merge_log = merge_chapter_sets(chapter_sets, plans)
    else:
        # Fallback: no chapter_sets (e.g., collect used old path); build empty merged chapters.
        # When no domains are marked active (e.g. plan merger didn't set it), fall back to all
        # domains so synthesis still produces content rather than an empty brief.
        primary_plan = plans[0]
        active_domains = _active_domains(primary_plan) or list(_DOMAINS)
        merged_chapters = [
            MergedChapter(domain=d, text=f"No data collected for {d}.")
            for d in active_domains
        ]
        merge_log = []

    # Assign each unique source one stable id across all domains, in
    # first-occurrence order — downstream entity filtering, synthesis prompts,
    # and the PDF's numbered Sources section all key off this shared registry.
    citation_registry = assign_global_citation_ids(merged_chapters)

    # ------------------------------------------------------------------
    # Step 2: Diversity recovery path
    # ------------------------------------------------------------------
    overlap = chapter_set_overlap(merged_chapters)
    if overlap > settings.tot.diversity_overlap_threshold and merged_chapters:
        all_warnings.append(
            f"Chapter set overlap {overlap:.2f} exceeds threshold "
            f"{settings.tot.diversity_overlap_threshold:.2f}; triggering diversity recovery."
        )
        logger.info("synthesize_node: diversity recovery triggered (overlap=%.2f)", overlap)

        all_plans_bus = get_all_plans_direct()
        survivor_ids = {p.get("plan_id", "") for p in plans}
        survivor_activations = [
            frozenset(d for d, v in p.get("domain_activations", {}).items() if v)
            for p in plans
        ]
        pruned = [
            p for p in all_plans_bus
            if not p.get("is_survivor", False) and p.get("plan_id", "") not in survivor_ids
        ]
        distinct_pruned = [
            p for p in pruned
            if frozenset(
                d for d, v in p.get("domain_activations", {}).items() if v
            ) not in survivor_activations
        ]
        distinct_pruned.sort(key=lambda p: p.get("combined_score", 0.0), reverse=True)

        if distinct_pruned:
            recovery_plan = distinct_pruned[0]
            recovery_drafts: list[ChapterDraft] = []
            # Run recovery agents concurrently, bounded by max_parallel_subagents.
            from agents import DOMAIN_AGENTS
            rec_sem = asyncio.Semaphore(settings.max_parallel_subagents)

            async def _run_recovery(domain: str) -> ChapterDraft | None:
                agent_cls = DOMAIN_AGENTS.get(domain)
                if agent_cls is None:
                    return None
                async with rec_sem:
                    try:
                        return await agent_cls().run(recovery_plan, state["run_id"])
                    except Exception as exc:
                        logger.warning("recovery: domain %s failed: %s", domain, exc)
                        return None

            rec_results = await asyncio.gather(
                *[_run_recovery(d) for d in _active_domains(recovery_plan)]
            )
            for draft in rec_results:
                if draft is not None:
                    recovery_drafts.append(draft)
                    usage_dicts.append(draft.usage)

            if recovery_drafts:
                supplementary_text = "\n\n".join(
                    f"**{d.domain.title()}**\n{d.text}" for d in recovery_drafts
                )
                supplementary_chapter = MergedChapter(
                    domain="supplementary",
                    text=supplementary_text,
                    citations=[c for d in recovery_drafts for c in d.citations],
                    source_plan_ids=[recovery_plan.get("plan_id", "recovery")],
                )
                merged_chapters.append(supplementary_chapter)
                # Extend (not replace) the registry so ids already used by other
                # chapters' prompts/markers stay stable.
                citation_registry = assign_global_citation_ids(
                    [supplementary_chapter], existing=citation_registry
                )
                logger.info("synthesize_node: appended supplementary section from recovery plan")

    # ------------------------------------------------------------------
    # Step 3: Synthesise each merged chapter → polished prose
    #
    # Entity-rich domains are decomposed into per-entity Tier-1 analyses
    # (subchapters), then rolled up into the domain chapter (Tier 2). Domains
    # with no discrete entities fall back to the legacy single-pass synthesis.
    # ------------------------------------------------------------------
    synth_agent = SynthesisAgent()
    masterdata = MasterDataService()
    primary_plan = plans[0] if plans else {}

    # Synthesize all domain chapters concurrently. The number of chapters in flight is
    # bounded by max_parallel_chapters; the actual LLM-call concurrency is capped
    # process-wide by crew_semaphore (settings.llm.max_concurrent_calls), so this can't
    # overwhelm the API key no matter how many domains there are. Each task accumulates
    # its own warnings/injection-flags/usages and the results are merged in order after
    # the gather, so chapters never race on shared state.
    chapter_sem = asyncio.Semaphore(settings.synthesis.max_parallel_chapters)

    async def _synth_chapter(mc: MergedChapter) -> dict:
        local_warnings: list[str] = []
        local_injection: list[str] = []
        local_usages: list[dict] = []
        chapter: dict | None = None
        async with chapter_sem:
            await log_event("progress", f"Writing the {friendly_domain(mc.domain)} chapter…")
            sub_question = f"{mc.domain} signals relevant to: {query}"
            try:
                # to_thread so a slow retrieval for one chapter doesn't block the others.
                raw_retrieved, stale = await asyncio.to_thread(
                    retriever.retrieve, sub_question, collection, settings.retrieval.top_k,
                )
                local_warnings.extend(w.message for w in stale)
            except Exception:
                raw_retrieved = []

            # Phase 9.1: Filter injection-tainted chunks before passing to synthesis
            clean_retrieved = []
            for chunk in raw_retrieved:
                chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
                warning = _guardrails.scan_for_injection(chunk_text)
                if warning:
                    flag = f"Chunk filtered ({mc.domain}): {warning}"
                    local_injection.append(flag)
                    logger.warning("synthesize_node: %s", flag)
                else:
                    clean_retrieved.append(chunk)

            if settings.synthesis.hierarchical_enabled:
                subdomains, enum_usage = await asyncio.to_thread(
                    enumerate_subdomains, mc.domain, mc, primary_plan, masterdata
                )
                if enum_usage:
                    local_usages.append(enum_usage)
            else:
                subdomains = []

            if len(subdomains) >= 2:
                # --- Tier 1: per-entity analyses (run concurrently, bounded) ---
                sem = asyncio.Semaphore(settings.synthesis.max_parallel_subdomains)

                async def _synth_one(sub):
                    evidence = assemble_entity_evidence(
                        sub, mc, retriever, collection, _guardrails, query
                    )
                    local_injection.extend(evidence.injection_flags)
                    await log_event("progress", f"Analyzing {sub.label} for {friendly_domain(mc.domain)}…")
                    async with sem:
                        return await asyncio.to_thread(
                            synth_agent.run_subchapter,
                            mc.domain,
                            sub.key,
                            sub.label,
                            evidence.figures,
                            evidence.datasets,
                            evidence.citations,
                            evidence.retrieved_chunks,
                            query,
                        )

                subchapters = list(await asyncio.gather(*[_synth_one(s) for s in subdomains]))

                if settings.synthesis.completeness_gate_enabled:
                    subchapters = await _remediate_subchapters(
                        mc, subdomains, subchapters, synth_agent, query,
                        retriever, collection, _guardrails, local_warnings,
                    )

                for sc in subchapters:
                    local_usages.append(sc.get("usage", {}))

                # --- Tier 2: roll the entity analyses up into the domain chapter ---
                rollup = await asyncio.to_thread(
                    synth_agent.run_rollup, mc.domain, mc, subchapters, query
                )
                local_usages.append(rollup.get("usage", {}))
                await log_event("progress", f"Finished the {friendly_domain(mc.domain)} chapter.")
                chapter = {
                    "domain": mc.domain,
                    "text": rollup["text"],
                    "figures": dict(mc.figures),
                    "datasets": mc.datasets,
                    "subchapters": subchapters,
                }
                logger.info(
                    "synthesize_node: %s decomposed into %d subchapters",
                    mc.domain, len(subchapters),
                )
            else:
                # Degenerate domain → legacy single-pass synthesis (no subchapters).
                result = await asyncio.to_thread(
                    synth_agent.run, mc.domain, mc, clean_retrieved, [], query
                )
                local_usages.append(result.get("usage", {}))
                result["datasets"] = mc.datasets
                result["subchapters"] = []

                if settings.synthesis.completeness_gate_enabled and completeness.is_fallback_text(
                    result.get("text", "")
                ):
                    # One resynthesis retry (benefits from synthesis_agent's retry/backoff);
                    # if it's still a placeholder, replace it with an honest message.
                    retry = await asyncio.to_thread(
                        synth_agent.run, mc.domain, mc, clean_retrieved, [], query
                    )
                    local_usages.append(retry.get("usage", {}))
                    if not completeness.is_fallback_text(retry.get("text", "")):
                        retry["datasets"] = mc.datasets
                        retry["subchapters"] = []
                        result = retry
                        local_warnings.append(f"completeness_gate: resynthesized {mc.domain} after initial gap")
                    else:
                        result["text"] = completeness.honest_fallback_message(mc.domain)
                        local_warnings.append(f"completeness_gate: {mc.domain} still incomplete after remediation")

                chapter = result
                await log_event("progress", f"Finished the {friendly_domain(mc.domain)} chapter.")
        return {
            "chapter": chapter, "warnings": local_warnings,
            "injection_flags": local_injection, "usages": local_usages,
        }

    chapter_results = await asyncio.gather(*[_synth_chapter(mc) for mc in merged_chapters])
    chapters: list[dict] = []
    for res in chapter_results:  # gather preserves merged_chapters order
        if res["chapter"] is not None:
            chapters.append(res["chapter"])
        all_warnings.extend(res["warnings"])
        all_injection_flags.extend(res["injection_flags"])
        usage_dicts.extend(res["usages"])

    # If all synthesis agents failed, fall back to raw merged chapter text so we always
    # produce a non-empty brief rather than an empty document.
    if not chapters and merged_chapters:
        logger.warning(
            "synthesize_node: all per-chapter synthesis failed; using raw merged text as fallback"
        )
        chapters = [
            {"domain": mc.domain, "text": mc.text, "figures": dict(mc.figures), "usage": {},
             "datasets": mc.datasets, "subchapters": []}
            for mc in merged_chapters
        ]

    # ------------------------------------------------------------------
    # Step 4: Executive summary
    # ------------------------------------------------------------------
    exec_summary = ""
    if chapters:
        await log_event("progress", "Writing the executive summary…")
        chapter_texts = "\n\n".join(
            f"## {ch['domain'].replace('_', ' ').title()}\n{ch['text']}"
            for ch in chapters
        )
        llm = LLMClient()
        target_min = settings.report.exec_summary_min_words
        target_max = settings.report.exec_summary_max_words
        try:
            msgs = exec_summary_messages(
                query, chapter_texts, min_words=target_min, max_words=target_max
            )
            # Run off the event loop and ride out transient 429s/timeouts, exactly
            # as the per-chapter synthesis calls do — otherwise this lone synchronous
            # call freezes the loop (stalling status polls) and dies on the first
            # rate-limit that follows the parallel subchapter/rollup burst.
            resp = await asyncio.to_thread(
                call_with_backoff, llm.complete, msgs, temperature=0.3
            )
            usage_dicts.append(llm_usage(resp.usage))
            exec_summary = resp.content or ""
            word_count = len(exec_summary.split())
            if not (target_min <= word_count <= target_max):
                msgs.append({"role": "assistant", "content": exec_summary})
                msgs.append({
                    "role": "user",
                    "content": (
                        f"The summary is {word_count} words. "
                        f"Revise it to be between {target_min} and {target_max} words."
                    ),
                })
                resp2 = await asyncio.to_thread(
                    call_with_backoff, llm.complete, msgs, temperature=0.3
                )
                usage_dicts.append(llm_usage(resp2.usage))
                exec_summary = resp2.content or exec_summary
        except Exception as exc:
            # Non-fatal: the chapters are still a valid report. But make it loud —
            # surface the reason in the warnings appendix instead of only logging it,
            # so the empty/fallback summary is explainable rather than mysterious.
            logger.warning("synthesize_node: exec summary generation failed (non-fatal): %s", exc)
            all_warnings.append(f"Executive summary generation failed after retries: {exc}")

    # Deterministic fallback: first 300 words of the first chapter so Gate 3 always
    # has content — but label it honestly so a truncated chapter chunk is never
    # presented as if it were a real executive summary.
    if not exec_summary and chapters:
        first_text = chapters[0].get("text", "") if isinstance(chapters[0], dict) else ""
        words = first_text.split()
        excerpt = " ".join(words[:300]) + ("…" if len(words) > 300 else "")
        exec_summary = (
            "_The executive summary could not be generated automatically for this run; "
            "the excerpt below is drawn from the first chapter as a placeholder._\n\n"
            f"{excerpt}"
        )

    # Guarantee the chat always shows something at completion, even when synthesis
    # produced no chapters at all (rather than silently leaving exec_summary empty).
    if not exec_summary and not chapters:
        exec_summary = "No content could be synthesized for this run — see warnings for details."

    cw = ContextWindow.from_state(
        state.get("context_messages", []), settings.react.run_token_budget
    )
    cw.add(
        "assistant",
        f"Synthesized {len(chapters)} domain chapters "
        f"(overlap={overlap:.2f}, merge_log={len(merge_log)} resolutions).",
    )

    store = SqliteStore()
    auto_approve = settings.gates.auto_approve_gates or bool(await store.get_preference("auto_approve_gates"))
    gate3_needed = settings.gates.gate_3_enabled and not auto_approve

    # Gate 3's interrupt() lives in its own node (gate3_node) rather than here,
    # since LangGraph replays a node from the top on resume — interrupting
    # after all this synthesis work would silently redo it on every approval.
    return {
        "synthesis_chapters": chapters,
        "merge_log": merge_log,
        "stage": "synthesize" if gate3_needed else "done",
        "context_messages": cw.to_state(),
        "warnings": all_warnings,
        "injection_flags": all_injection_flags,
        "exec_summary": exec_summary,
        "citation_registry": citation_registry,
        **accumulate(state, *usage_dicts),
    }


async def gate3_node(state: AgentState) -> dict:
    """Gate 3 — human review of the synthesized intelligence brief.

    Split out of synthesize_node so that approving the gate doesn't replay the
    expensive synthesis pipeline (LangGraph re-runs a node from the top on
    resume). This node only re-reads the already-computed chapters from state.
    """
    chapters = state.get("synthesis_chapters") or []
    sections = [
        {
            "title": ch["domain"].replace("_", " ").title(),
            "content": ch["text"][:800],
            "subsections": [
                {
                    "title": sc.get("subdomain_label", sc.get("subdomain_key", "")),
                    "content": sc.get("text", "")[:600],
                }
                for sc in (ch.get("subchapters") or [])
            ],
        }
        for ch in chapters
    ]
    gate3_payload = {
        "gate": 3,
        "sections": sections,
        "executive_summary": state.get("exec_summary", ""),
        "warnings": state.get("warnings", []),
    }
    decision = interrupt(gate3_payload)
    if decision == "redirect":
        return {"synthesis_chapters": [], "exec_summary": "", "stage": "synthesize"}
    return {"stage": "done"}


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def understand_router(state: AgentState) -> str:
    stage = state.get("stage")
    if stage in ("clarification_needed", "error", "partial"):
        return END
    if stage == "understand":
        return "gate1"   # plan ready, gate 1 review pending
    return "collect"


def gate1_router(state: AgentState) -> str:
    # Gate 1 redirect sets stage="understand" to re-plan; approve sets "collect".
    return "understand" if state.get("stage") == "understand" else "collect"


def synthesize_router(state: AgentState) -> str:
    if state.get("stage") == "partial":
        return "partial_brief"   # soft-timeout stop mid-synthesis → real partial report
    if state.get("stage") == "synthesize":
        return "gate3"   # brief ready, gate 3 review pending
    return END


def gate3_router(state: AgentState) -> str:
    # Gate 3 redirect sets stage="synthesize" to re-synthesize; approve sets "done".
    return "synthesize" if state.get("stage") == "synthesize" else END


def react_router(state: AgentState) -> str:
    if state.get("stage") == "error":
        return "synthesize"   # surface the failure; do not retry-loop
    if state.get("stage") == "partial":
        return "partial_brief"
    # Silent retries are bounded by collect_max_retries (Gate 2 is NOT re-shown during
    # them); max_iterations remains the outer safety cap.
    retries_exhausted = (
        state["react_iterations"] >= settings.react.collect_max_retries
        or state["react_iterations"] >= settings.react.max_iterations
    )
    if state["confidence"] >= settings.react.confidence_threshold or retries_exhausted:
        return "data_review"   # data settled → show Gate 2 once (gaps flagged if any)
    return "backtrack"         # silent retry


def data_review_router(state: AgentState) -> str:
    # Gate 2 redirect sets stage="collect" to re-collect; approve sets "synthesize".
    return "collect" if state.get("stage") == "collect" else "synthesize"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

_graph = StateGraph(AgentState)
_graph.add_node("understand", understand_node)
_graph.add_node("gate1", gate1_node)
_graph.add_node("collect", collect_node)
_graph.add_node("backtrack", backtrack_node)
_graph.add_node("data_review", data_review_node)
_graph.add_node("synthesize", synthesize_node)
_graph.add_node("gate3", gate3_node)
_graph.add_node("partial_brief", partial_brief_node)
_graph.set_entry_point("understand")
_graph.add_conditional_edges(
    "understand",
    understand_router,
    {"gate1": "gate1", "collect": "collect", END: END},
)
_graph.add_conditional_edges(
    "gate1",
    gate1_router,
    {"understand": "understand", "collect": "collect"},
)
_graph.add_conditional_edges(
    "collect",
    react_router,
    {"data_review": "data_review", "backtrack": "backtrack",
     "synthesize": "synthesize", "partial_brief": "partial_brief"},
)
_graph.add_edge("backtrack", "collect")
_graph.add_conditional_edges(
    "data_review",
    data_review_router,
    {"synthesize": "synthesize", "collect": "collect"},
)
_graph.add_conditional_edges(
    "synthesize",
    synthesize_router,
    {"gate3": "gate3", "partial_brief": "partial_brief", END: END},
)
_graph.add_conditional_edges(
    "gate3",
    gate3_router,
    {"synthesize": "synthesize", END: END},
)
_graph.add_edge("partial_brief", END)

compiled = _graph.compile(checkpointer=MemorySaver())


async def init_checkpointer() -> None:
    """Replace the in-memory checkpointer with a durable SQLite-backed one.

    Call once at app startup (inside the running event loop, since aiosqlite
    needs it). Without this, a dev-server autoreload between gate pauses wipes
    in-flight run state and resuming a gate raises KeyError on the next state
    read (e.g. 'user_query').
    """
    global compiled, _checkpoint_conn
    checkpoint_path = os.path.expanduser(settings.stores.checkpoint_path)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    _checkpoint_conn = aiosqlite.connect(checkpoint_path)
    await _checkpoint_conn
    checkpointer = AsyncSqliteSaver(_checkpoint_conn)
    await checkpointer.setup()
    compiled = _graph.compile(checkpointer=checkpointer)
