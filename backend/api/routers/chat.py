import asyncio
import logging
import time
import uuid
from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas.chat import ChatRequest, ChatResponse
from config import settings
from core.event_logger import log_event, set_run_context
from core import graph as graph_module
from core.graph import AgentState
from memory.sqlite_store import SqliteStore
from langgraph.types import Command

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

_NODE_MESSAGES: dict[str, str] = {
    "understand":    "Analyzing your question and building research plans…",
    "collect":       "Collecting intelligence from data sources…",
    "backtrack":     "Refining research approach…",
    "synthesize":    "Synthesizing findings into intelligence brief…",
    "partial_brief": "Assembling available findings…",
}


@router.post("/chat", response_model=ChatResponse)
async def start_chat(body: ChatRequest, background_tasks: BackgroundTasks) -> ChatResponse:
    run_id = str(uuid.uuid4())
    session_id = body.session_id or str(uuid.uuid4())

    store = SqliteStore()
    if settings.stores.wipe_session_stores_on_chat:
        await store.wipe_session(session_id)

    initial_state: AgentState = {
        "run_id": run_id,
        "session_id": session_id,
        "user_query": body.query,
        "plans": [],
        "collection_manifest": {},
        "chapter_sets": {},
        "synthesis_chapters": [],
        "merge_log": [],
        "confidence": 0.0,
        "react_iterations": 0,
        "context_messages": [],
        "stage": "understand",
        "warnings": [],
        "error": None,
        "cumulative_cost_usd": 0.0,
        "api_call_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "injection_flags": [],
        "clarification_done": False,
        "exec_summary": "",
        "run_start_time": time.time(),
        "paused_seconds": 0.0,
        "timeout_prompt_count": 0,
    }
    await store.upsert_run(
        run_id, session_id, body.query, "running", "understand",
        initial_state=initial_state,
    )
    background_tasks.add_task(_run_graph, run_id, session_id, body.query, initial_state)
    return ChatResponse(run_id=run_id, session_id=session_id, status="running")


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    row = await SqliteStore().get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return row


@router.post("/chat/stream")
async def stream_chat(body: ChatRequest):
    raise NotImplementedError


async def _persist_stream_event(event, run_id, session_id, query, store) -> None:
    """Handle one astream event: update the status message on node start and the
    live cost/token counter on node completion."""
    etype = event["event"]
    if etype == "on_chain_start":
        node = event.get("metadata", {}).get("langgraph_node") or event.get("name", "")
        # LangGraph sometimes emits namespace-prefixed names like "ns:collect" — strip the prefix.
        node = node.split(":")[-1] if ":" in node else node
        msg = _NODE_MESSAGES.get(node)
        if msg:
            await log_event("progress", msg, stage=node)
            await store.upsert_run(
                run_id, session_id, query,
                status="running", stage=node,
                status_message=msg,
            )
    elif etype == "on_chain_end":
        output = event.get("data", {}).get("output")
        if isinstance(output, dict) and "cumulative_cost_usd" in output:
            await store.upsert_run(
                run_id, session_id, query,
                status="running", stage=output.get("stage", "") or "",
                cumulative_cost_usd=output.get("cumulative_cost_usd", 0.0),
                api_call_count=output.get("api_call_count", 0),
                total_tokens=output.get("total_tokens", 0),
            )


def _manifest_to_sources(manifest: dict) -> list[dict]:
    """Flatten the per-domain collection manifest into the Sources-panel list.

    Entries are already produced in the enriched shape by graph._draft_source_entries
    (typed datasets + failed tools), so this just concatenates them across domains.
    """
    sources: list[dict] = []
    for items in (manifest or {}).values():
        sources.extend(items or [])
    return sources


def _assemble_brief(chapters: list[dict]) -> str:
    """Render synthesized chapters as hierarchical markdown.

    Entity-rich domains get a heading per domain, a sub-heading + body per
    entity subchapter, then the domain landscape summary. Degenerate domains
    render as a single domain section (same content as before).
    """
    parts: list[str] = []
    for ch in chapters:
        domain_title = ch.get("domain", "").replace("_", " ").title()
        subchapters = ch.get("subchapters") or []
        if subchapters:
            parts.append(f"# {domain_title}")
            for sc in subchapters:
                label = sc.get("subdomain_label") or sc.get("subdomain_key") or "Entity"
                parts.append(f"## {label}\n\n{sc.get('text', '')}")
            parts.append(f"## {domain_title} — Landscape Summary\n\n{ch.get('text', '')}")
        else:
            parts.append(f"# {domain_title}\n\n{ch.get('text', '')}")
    return "\n\n".join(parts)


async def _handle_graph_result(
    final: dict,
    run_id: str,
    session_id: str,
    query: str,
) -> bool:
    """Process an ainvoke result. Returns True when the run is fully complete."""
    config = {"configurable": {"thread_id": run_id}}
    store = SqliteStore()

    # Check for gate interrupt (graph suspended, more nodes pending)
    snapshot = await graph_module.compiled.aget_state(config)
    if snapshot.next:
        pending = [intr for task in snapshot.tasks for intr in task.interrupts]
        if pending:
            gate_data = pending[0].value

            if gate_data.get("type") == "timeout_check":
                await store.upsert_run(
                    run_id, session_id, query,
                    status="waiting_timeout_confirm",
                    stage=final.get("stage", ""),
                    gate_data=gate_data,
                    paused_at=time.time(),
                )
                return False

            gate_num = gate_data.get("gate")

            extra_kwargs: dict = {}
            if gate_num == 2:
                # Persist sources at Gate 2 so the right panel populates during review
                extra_kwargs["sources"] = _manifest_to_sources(
                    final.get("collection_manifest", {})
                )

            await store.upsert_run(
                run_id, session_id, query,
                status=f"waiting_gate_{gate_num}",
                stage=final.get("stage", "understand"),
                gate_data=gate_data,
                paused_at=time.time(),
                **extra_kwargs,
            )
            return False

    # Normal / clarification / error completion
    final_stage = final.get("stage", "done")
    if final_stage == "clarification_needed":
        await store.upsert_run(
            run_id, session_id, query,
            status="waiting_clarification",
            stage="clarification_needed",
            error=final.get("error"),
        )
        return True

    if final_stage == "error":
        await store.upsert_run(
            run_id, session_id, query,
            status="error",
            stage="error",
            error=final.get("error") or "Pipeline error — check backend logs for details.",
        )
        return True

    sources = _manifest_to_sources(final.get("collection_manifest", {}))

    brief = _assemble_brief(final.get("synthesis_chapters", []))

    # Defensive backstop: a run can reach here with no exception raised yet still
    # have produced nothing committed to state — e.g. synthesize_node finished its
    # work in local variables but was interrupted (Gate 3 / soft timeout) before
    # its final return, then exited via a redirect/partial route that returned
    # empty synthesis_chapters. Before giving up, try to reconstruct a brief from
    # chapter_sets (committed by collect_node, so it survives that loss).
    if not (final.get("synthesis_chapters") or final.get("exec_summary") or brief):
        recovered = [
            {"domain": draft.get("domain")
                       or (key.split("::")[-1] if "::" in key else key),
             "text": draft.get("text", "")}
            for key, draft in (final.get("chapter_sets") or {}).items()
            if (draft.get("text") or "").strip()
        ]
        recovered_brief = _assemble_brief(recovered)
        if recovered_brief:
            logger.warning(
                "Run %s reached completion with no synthesis_chapters/exec_summary; "
                "recovered a brief from %d chapter_sets (stage=%s).",
                run_id, len(recovered), final_stage,
            )
            final["synthesis_chapters"] = recovered
            final["warnings"] = [
                *final.get("warnings", []),
                "Report reconstructed from collected data: synthesis was "
                "interrupted before it could finalize, so chapters are shown "
                "without the executive summary or final polish.",
            ]
            brief = recovered_brief
        else:
            logger.warning(
                "Run %s ended empty: stage=%s, synthesis_chapters=%s, "
                "exec_summary=%s, chapter_sets=%s.",
                run_id, final_stage,
                bool(final.get("synthesis_chapters")),
                bool(final.get("exec_summary")),
                bool(final.get("chapter_sets")),
            )
            await store.upsert_run(
                run_id, session_id, query,
                status="error",
                stage=final_stage,
                error=(
                    "The run ended without producing a report — it may have been "
                    "stopped or redirected before synthesis completed. Please retry."
                ),
                warnings=final.get("warnings", []),
            )
            return True

    await store.upsert_run(
        run_id, session_id, query,
        status="done",
        stage=final_stage,
        confidence=final.get("confidence", 0.0),
        brief=brief,
        sources=sources,
        gate_data=None,  # clear any stale gate data now that the run is done
        exec_summary=final.get("exec_summary", ""),
        warnings=final.get("warnings", []),
        injection_flags=final.get("injection_flags", []),
        merge_log=final.get("merge_log", []),
        cumulative_cost_usd=final.get("cumulative_cost_usd", 0.0),
        api_call_count=final.get("api_call_count", 0),
        total_tokens=final.get("total_tokens", 0),
        plans=final.get("plans", []),
    )

    if final.get("synthesis_chapters"):
        try:
            import logging as _logging
            from reports.assembler import Assembler
            from reports.pdf_generator import PdfGenerator
            draft = Assembler.assemble(final, run_id=run_id, query=query)
            PdfGenerator.generate(draft)
            _logging.getLogger(__name__).info("PDF report generated for run %s", run_id)
        except Exception as pdf_exc:
            logger.exception("PDF generation failed (non-fatal)")
            existing_warnings = final.get("warnings", [])
            await store.upsert_run(
                run_id, session_id, query,
                status="done",
                stage=final_stage,
                warnings=[*existing_warnings, f"PDF generation failed: {pdf_exc}"],
            )

    return True


async def _run_graph(
    run_id: str,
    session_id: str,
    query: str,
    initial_state: AgentState,
) -> None:
    config = {"configurable": {"thread_id": run_id}}
    store = SqliteStore()
    set_run_context(run_id, stage="understand")

    async def _stream() -> None:
        async for event in graph_module.compiled.astream_events(initial_state, config, version="v2"):
            await _persist_stream_event(event, run_id, session_id, query, store)

    try:
        await asyncio.wait_for(_stream(), timeout=settings.react.hard_time_limit_s)
        snapshot = await graph_module.compiled.aget_state(config)
        final = snapshot.values
        await _handle_graph_result(final, run_id, session_id, query)
    except asyncio.TimeoutError:
        await store.upsert_run(
            run_id, session_id, query,
            status="timeout", stage="unknown",
            error=f"Run exceeded hard time limit of {settings.react.hard_time_limit_s}s",
        )
    except Exception as exc:
        logger.exception("_run_graph failed for run %s", run_id)
        await store.upsert_run(
            run_id, session_id, query, status="error", stage="error", error=str(exc)
        )


async def _resume_graph(
    run_id: str,
    session_id: str,
    query: str,
    decision: str,
) -> None:
    """Resume a graph that is suspended at a gate interrupt."""
    config = {"configurable": {"thread_id": run_id}}
    store = SqliteStore()
    existing_row = await store.get_run(run_id)
    set_run_context(run_id)

    # Exclude the time spent waiting for this decision from the soft-timeout clock.
    paused_at = existing_row.get("paused_at") if existing_row else None
    if paused_at:
        pause_duration = max(0.0, time.time() - paused_at)
        snapshot = await graph_module.compiled.aget_state(config)
        values = snapshot.values or {}
        prior_paused = values.get("paused_seconds", 0.0)
        update = {"paused_seconds": prior_paused + pause_duration}
        # Bump the rolling timeout threshold so the next soft-timeout prompt is
        # due another soft_timeout_s later, instead of re-firing on the next node.
        if existing_row.get("status") == "waiting_timeout_confirm" and decision == "approve":
            update["timeout_prompt_count"] = values.get("timeout_prompt_count", 0) + 1
        await graph_module.compiled.aupdate_state(config, update)

    async def _stream() -> None:
        async for event in graph_module.compiled.astream_events(Command(resume=decision), config, version="v2"):
            await _persist_stream_event(event, run_id, session_id, query, store)

    try:
        await asyncio.wait_for(_stream(), timeout=settings.react.hard_time_limit_s)
        snapshot = await graph_module.compiled.aget_state(config)
        final = snapshot.values
        await _handle_graph_result(final, run_id, session_id, query)
    except asyncio.TimeoutError:
        await store.upsert_run(
            run_id, session_id, query,
            status="timeout", stage="unknown",
            error=f"Run exceeded hard time limit of {settings.react.hard_time_limit_s}s",
        )
    except Exception as exc:
        logger.exception("_resume_graph failed for run %s", run_id)
        await store.upsert_run(
            run_id, session_id, query, status="error", stage="error", error=str(exc)
        )
