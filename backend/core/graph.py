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
from models.usage import accumulate, llm_usage, merge_usage
from prompts.synthesize_prompt import exec_summary_messages
from core.event_logger import log_event
from core.friendly_names import friendly_domain
from core.tot.proposer import PlanProposer
from core.tot.scorer import score_and_prune
from core.tot.schemas import ResearchContext
from core.research_agent import ResearchAgent
from core.plan_merger import PlanMerger
from agents.grounding_agent import GroundingAgent
from state_bus.server import clear_plans, write_plan_direct, get_all_plans_direct
from memory.context_window import ContextWindow
from memory.sqlite_store import SqliteStore
from core.merger import merge_chapter_sets, chapter_set_overlap
from core.schemas import ChapterDraft, MergedChapter
from core.subdomains import enumerate_subdomains, assemble_entity_evidence
from core import completeness
from core.tool_router import async_route
from services.masterdata_service import MasterDataService
from agents.synthesis_agent import SynthesisAgent
from retrieval import Retriever
from retrieval.chunker import Chunker

logger = logging.getLogger(__name__)

_DOMAINS = [
    "competition",
    "distributors",
    "customers",
    "mining_projects",
    "commodities",
    "macro_geopolitics",
    "general_search",
]

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
_REQUIRED_ENTITY_PREFS = ("equipment_models", "mine_sites", "competitor_tickers")


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
    # Soft-timeout tracking: unix timestamp set once at graph start
    run_start_time: float
    # Cumulative seconds spent paused at gates/timeout-checks, excluded from elapsed-time math
    paused_seconds: float


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


_collect_locks: dict[str, asyncio.Lock] = {}


def _collect_lock(run_id: str) -> asyncio.Lock:
    lock = _collect_locks.get(run_id)
    if lock is None:
        lock = _collect_locks[run_id] = asyncio.Lock()
    return lock


async def _persist_partial_sources(
    run_id: str, session_id: str, query: str, domain: str, draft: "ChapterDraft"
) -> None:
    """Append one domain's sources to the run row as soon as its agent finishes,
    so the frontend Sources panel can populate live during collection instead of
    waiting for the whole collect_node (and Gate 2) to complete."""
    new_entries = [
        {
            "domain": domain,
            "tool": "WebSearch",
            "title": url,
            "url": url,
            "published_at": None,
        }
        for url in (draft.citations or [])
    ]
    if not new_entries:
        return
    store = SqliteStore()
    async with _collect_lock(run_id):
        row = await store.get_run(run_id)
        existing = list(row.get("sources") or []) if row else []
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

    # Write chapter text to the per-run Chroma collection (unstructured store)
    if draft.text.strip():
        try:
            docs = chunker.chunk_text(draft.text, source=f"{domain}_agent", domain=domain)
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
    if elapsed <= settings.react.soft_timeout_s:
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
    try:
        _timeout_interrupt_if_needed(state)
    except _TimeoutStopSignal:
        return {"stage": "partial"}

    query = state["user_query"]

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
                "understand_node: research found %d companies, %d tickers, %d commodities",
                len(research_context.companies),
                len(research_context.tickers),
                len(research_context.commodities),
            )
            await log_event(
                "progress",
                f"Found {len(research_context.companies)} relevant companies and "
                f"{len(research_context.commodities)} commodities to investigate.",
            )
        except Exception as exc:
            logger.warning("understand_node: ResearchAgent failed (%s) — proceeding without research", exc)

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

    # 4. Ground to depth-2 via CrewAI critic
    await log_event("progress", "Reviewing and refining the research plans…")
    grounder_usage: dict = {}
    try:
        grounder = GroundingAgent()
        grounded = grounder.run(candidates)
        grounder_usage = grounder.last_usage
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
    merger_usage: dict = {}
    try:
        merger = PlanMerger()
        consolidated = await merger.merge(survivors, research_context, run_id=state["run_id"])
        merger_usage = merger.last_usage
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

    if settings.gates.gate_1_enabled and not settings.gates.auto_approve_gates:
        gate1_payload = {
            "gate": 1,
            "plan": consolidated.model_dump(),
        }
        decision = interrupt(gate1_payload)
        if decision == "redirect":
            return {"plans": [], "stage": "understand", **usage_delta}

    # Pass the consolidated plan forward. collect_node iterates `plans` as a list;
    # wrapping in a list of one preserves backward compatibility.
    return {
        "plans": [consolidated.model_dump()],
        "stage": "collect",
        "context_messages": cw.to_state(),
        **usage_delta,
    }


async def collect_node(state: AgentState) -> dict:
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
            state["run_id"], state["session_id"], state["user_query"], domain, draft
        )

    # Build the sources manifest from the full merged chapter_sets (carried + new), so the
    # sources panel reflects everything collected across iterations.
    manifest: dict[str, list] = {d: [] for d in _DOMAINS}  # backward-compat
    for key, draft in chapter_sets.items():
        plan_id, _, domain = key.partition("::")
        citations = draft.get("citations") or []
        if citations:
            for url in citations:
                manifest.setdefault(domain, []).append({
                    "title": url,
                    "url": url,
                    "_tool": "web_search",
                    "published_at": None,
                })
        else:
            manifest.setdefault(domain, []).append({
                "title": f"{domain}/{plan_id[:8]}",
                "description": (draft.get("text") or "")[:200],
            })

    if collected_tool_errors:
        logger.warning("collect_node: %d tool/agent failures: %s",
                       len(collected_tool_errors), collected_tool_errors)

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

    Surfaces the actual datasets collected per domain (tables / lists / summaries)
    plus any tool failures as gap flags. On "redirect" the collection is reset and
    re-run; on approve the graph proceeds to synthesis.
    """
    if not settings.gates.gate_2_enabled or settings.gates.auto_approve_gates:
        return {"stage": "synthesize"}

    chapter_sets: dict[str, dict] = dict(state.get("chapter_sets") or {})
    domains_payload = []
    for domain in _DOMAINS:
        drafts = [v for k, v in chapter_sets.items() if k.endswith(f"::{domain}")]
        if not drafts:
            continue
        datasets: list[dict] = []
        errors: list[str] = []
        for d in drafts:
            datasets.extend(d.get("datasets") or [])
            errors.extend(d.get("tool_errors") or [])
        if not datasets and not errors:
            continue
        domains_payload.append({"domain": domain, "datasets": datasets, "errors": errors})

    decision = interrupt({"gate": 2, "domains": domains_payload})
    if decision == "redirect":
        return {"chapter_sets": {}, "confidence": 0.0, "react_iterations": 0, "stage": "collect"}
    return {"stage": "synthesize"}


async def backtrack_node(state: AgentState) -> dict:
    return {"react_iterations": state["react_iterations"] + 1}


async def partial_brief_node(state: AgentState) -> dict:
    """Assemble a partial brief from whatever chapter_sets exist when a guardrail fires."""
    chapter_sets = state.get("chapter_sets") or {}
    all_warnings = list(state.get("warnings", []))

    chapters: list[dict] = []
    synth_usages: list[dict] = []
    if chapter_sets:
        synth_agent = SynthesisAgent()
        for key, draft_dict in chapter_sets.items():
            domain = draft_dict.get("domain") or (key.split("::")[-1] if "::" in key else key)
            mc = MergedChapter(domain=domain, text=draft_dict.get("text", ""))
            try:
                result = synth_agent.run(domain, mc, [], [])
                synth_usages.append(result.get("usage", {}))
                chapters.append(result)
            except Exception as exc:
                logger.warning("partial_brief_node: synthesis failed for %s: %s", domain, exc)
                chapters.append({"domain": domain, "text": mc.text})
    else:
        all_warnings.append("Partial brief: no data was collected before the guardrail fired.")

    return {
        "synthesis_chapters": chapters,
        "stage": "done",
        "warnings": all_warnings,
        **accumulate(state, *synth_usages),
    }


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
                citations = list(dict.fromkeys(citations + evidence.citations))
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
                            citations = list(dict.fromkeys(
                                citations + [i["url"] for i in items if i.get("url")]
                            ))
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
    try:
        _timeout_interrupt_if_needed(state)
    except _TimeoutStopSignal:
        return {"stage": "partial"}

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
            # Run recovery agents sequentially (not parallel — this is a recovery path)
            from agents import DOMAIN_AGENTS
            for domain in _active_domains(recovery_plan):
                agent_cls = DOMAIN_AGENTS.get(domain)
                if agent_cls is None:
                    continue
                try:
                    draft = await agent_cls().run(recovery_plan, state["run_id"])
                    recovery_drafts.append(draft)
                    usage_dicts.append(draft.usage)
                except Exception as exc:
                    logger.warning("recovery: domain %s failed: %s", domain, exc)

            if recovery_drafts:
                supplementary_text = "\n\n".join(
                    f"**{d.domain.title()}**\n{d.text}" for d in recovery_drafts
                )
                merged_chapters.append(MergedChapter(
                    domain="supplementary",
                    text=supplementary_text,
                    source_plan_ids=[recovery_plan.get("plan_id", "recovery")],
                ))
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
    chapters: list[dict] = []

    for mc in merged_chapters:
        await log_event("progress", f"Writing the {friendly_domain(mc.domain)} chapter…")
        sub_question = f"{mc.domain} signals relevant to: {query}"
        try:
            raw_retrieved, stale = retriever.retrieve(
                sub_question,
                collection,
                top_k=settings.retrieval.top_k,
            )
            all_warnings.extend(w.message for w in stale)
        except Exception:
            raw_retrieved = []

        # Phase 9.1: Filter injection-tainted chunks before passing to synthesis
        clean_retrieved = []
        for chunk in raw_retrieved:
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            warning = _guardrails.scan_for_injection(chunk_text)
            if warning:
                flag = f"Chunk filtered ({mc.domain}): {warning}"
                all_injection_flags.append(flag)
                logger.warning("synthesize_node: %s", flag)
            else:
                clean_retrieved.append(chunk)

        if settings.synthesis.hierarchical_enabled:
            subdomains, enum_usage = enumerate_subdomains(mc.domain, mc, primary_plan, masterdata)
            if enum_usage:
                usage_dicts.append(enum_usage)
        else:
            subdomains = []

        if len(subdomains) >= 2:
            # --- Tier 1: per-entity analyses (run concurrently, bounded) ---
            sem = asyncio.Semaphore(settings.synthesis.max_parallel_subdomains)

            async def _synth_one(sub):
                evidence = assemble_entity_evidence(
                    sub, mc, retriever, collection, _guardrails, query
                )
                all_injection_flags.extend(evidence.injection_flags)
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

            subchapters = await asyncio.gather(*[_synth_one(s) for s in subdomains])
            subchapters = list(subchapters)

            if settings.synthesis.completeness_gate_enabled:
                subchapters = await _remediate_subchapters(
                    mc, subdomains, subchapters, synth_agent, query,
                    retriever, collection, _guardrails, all_warnings,
                )

            for sc in subchapters:
                usage_dicts.append(sc.get("usage", {}))

            # --- Tier 2: roll the entity analyses up into the domain chapter ---
            rollup = synth_agent.run_rollup(mc.domain, mc, subchapters, query)
            usage_dicts.append(rollup.get("usage", {}))
            await log_event("progress", f"Finished the {friendly_domain(mc.domain)} chapter.")
            chapters.append({
                "domain": mc.domain,
                "text": rollup["text"],
                "figures": dict(mc.figures),
                "datasets": mc.datasets,
                "subchapters": subchapters,
            })
            logger.info(
                "synthesize_node: %s decomposed into %d subchapters",
                mc.domain, len(subchapters),
            )
        else:
            # Degenerate domain → legacy single-pass synthesis (no subchapters).
            result = synth_agent.run(mc.domain, mc, clean_retrieved, [], query)
            usage_dicts.append(result.get("usage", {}))
            result["datasets"] = mc.datasets
            result["subchapters"] = []

            if settings.synthesis.completeness_gate_enabled and completeness.is_fallback_text(
                result.get("text", "")
            ):
                # One resynthesis retry (benefits from synthesis_agent's retry/backoff);
                # if it's still a placeholder, replace it with an honest message.
                retry = synth_agent.run(mc.domain, mc, clean_retrieved, [], query)
                usage_dicts.append(retry.get("usage", {}))
                if not completeness.is_fallback_text(retry.get("text", "")):
                    retry["datasets"] = mc.datasets
                    retry["subchapters"] = []
                    result = retry
                    all_warnings.append(f"completeness_gate: resynthesized {mc.domain} after initial gap")
                else:
                    result["text"] = completeness.honest_fallback_message(mc.domain)
                    all_warnings.append(f"completeness_gate: {mc.domain} still incomplete after remediation")

            chapters.append(result)
            await log_event("progress", f"Finished the {friendly_domain(mc.domain)} chapter.")

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
            resp = llm.complete(msgs, temperature=0.3)
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
                resp2 = llm.complete(msgs, temperature=0.3)
                usage_dicts.append(llm_usage(resp2.usage))
                exec_summary = resp2.content or exec_summary
        except Exception as exc:
            logger.warning("synthesize_node: exec summary generation failed (non-fatal): %s", exc)

    # Deterministic fallback: first 300 words of the first chapter so Gate 3 always has content.
    if not exec_summary and chapters:
        first_text = chapters[0].get("text", "") if isinstance(chapters[0], dict) else ""
        words = first_text.split()
        exec_summary = " ".join(words[:300]) + ("…" if len(words) > 300 else "")

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

    if settings.gates.gate_3_enabled and not settings.gates.auto_approve_gates:
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
            "executive_summary": exec_summary,
            "warnings": all_warnings,
        }
        decision = interrupt(gate3_payload)
        if decision == "redirect":
            return {
                "synthesis_chapters": [], "exec_summary": "", "stage": "synthesize",
                **accumulate(state, *usage_dicts),
            }

    return {
        "synthesis_chapters": chapters,
        "merge_log": merge_log,
        "stage": "done",
        "context_messages": cw.to_state(),
        "warnings": all_warnings,
        "injection_flags": all_injection_flags,
        "exec_summary": exec_summary,
        **accumulate(state, *usage_dicts),
    }


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

def understand_router(state: AgentState) -> str:
    stage = state.get("stage")
    if stage in ("clarification_needed", "error", "partial"):
        return END
    if stage == "understand":
        return "understand"   # gate 1 redirect self-loop
    return "collect"


def synthesize_router(state: AgentState) -> str:
    if state.get("stage") == "partial":
        return "partial_brief"   # soft-timeout stop mid-synthesis → real partial report
    if state.get("stage") == "synthesize":
        return "synthesize"   # gate 3 redirect self-loop
    return END


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
_graph.add_node("collect", collect_node)
_graph.add_node("backtrack", backtrack_node)
_graph.add_node("data_review", data_review_node)
_graph.add_node("synthesize", synthesize_node)
_graph.add_node("partial_brief", partial_brief_node)
_graph.set_entry_point("understand")
_graph.add_conditional_edges(
    "understand",
    understand_router,
    {"collect": "collect", "understand": "understand", END: END},
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
    {"synthesize": "synthesize", "partial_brief": "partial_brief", END: END},
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
