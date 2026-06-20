import asyncio
import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.schemas.gate import (
    ClarificationAnswer,
    GateDecisionResponse,
)
from config import settings
from memory.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["gates"])


def _episodic_summary(run: dict, plan: dict | None) -> str:
    """Build the one-chunk episodic document from a finished run + its winning plan."""
    lines = [
        "EPISODIC MEMORY — past successful research report.",
        f"Query: {run.get('query') or ''}",
        "",
        "Executive summary:",
        (run.get("exec_summary") or run.get("brief") or "")[:4000],
    ]
    if plan:
        active = [d for d, on in (plan.get("domain_activations") or {}).items() if on]
        tools = [tc.get("tool", "") for tc in (plan.get("tool_calls") or [])]
        lines += [
            "",
            f"Winning plan ({plan.get('plan_id', 'unknown')}):",
            f"Domains: {', '.join(active)}",
            f"Rationale: {plan.get('rationale', '')}",
            f"Tools used: {', '.join(t for t in tools if t)}",
        ]
    return "\n".join(lines)


# ── Gate data read ──────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/gates/{gate}")
async def get_gate(run_id: str, gate: int) -> dict:
    store = SqliteStore()
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") != f"waiting_gate_{gate}":
        raise HTTPException(
            status_code=409,
            detail=f"Run is not paused at gate {gate} (current status: {run.get('status')})",
        )
    return {"run_id": run_id, "gate": gate, **(run.get("gate_data") or {})}


# ── Gate approve ────────────────────────────────────────────────────────────

@router.post("/runs/{run_id}/gates/{gate}/approve", response_model=GateDecisionResponse)
async def approve_gate(
    run_id: str,
    gate: int,
    body: dict,
    background_tasks: BackgroundTasks,
) -> GateDecisionResponse:
    from api.routers.chat import _resume_graph  # delayed import avoids circular dependency

    store = SqliteStore()
    run = await store.get_run(run_id)
    if run is None or run.get("status") != f"waiting_gate_{gate}":
        raise HTTPException(
            status_code=404,
            detail=f"Run not found or not paused at gate {gate}",
        )

    session_id = run.get("session_id", "")
    query = run.get("query", "")
    await store.upsert_run(
        run_id, session_id, query,
        status="running",
        stage=run.get("stage", ""),
    )
    background_tasks.add_task(_resume_graph, run_id, session_id, query, "approve")
    return GateDecisionResponse(run_id=run_id, gate=gate, decision="approve", next_status="running")


# ── Gate redirect ───────────────────────────────────────────────────────────

@router.post("/runs/{run_id}/gates/{gate}/redirect", response_model=GateDecisionResponse)
async def redirect_gate(
    run_id: str,
    gate: int,
    body: dict,
    background_tasks: BackgroundTasks,
) -> GateDecisionResponse:
    from api.routers.chat import _resume_graph  # delayed import avoids circular dependency

    store = SqliteStore()
    run = await store.get_run(run_id)
    if run is None or run.get("status") != f"waiting_gate_{gate}":
        raise HTTPException(
            status_code=404,
            detail=f"Run not found or not paused at gate {gate}",
        )

    session_id = run.get("session_id", "")
    query = run.get("query", "")
    redirect_stage = {1: "understand", 2: "collect", 3: "synthesize"}.get(gate, "understand")
    await store.upsert_run(
        run_id, session_id, query,
        status="running",
        stage=redirect_stage,
    )
    background_tasks.add_task(_resume_graph, run_id, session_id, query, "redirect")
    return GateDecisionResponse(run_id=run_id, gate=gate, decision="redirect", next_status="running")


# ── Episodic memory save (manual "high-quality research" action) ─────────────

@router.post("/runs/{run_id}/episodic/save")
async def save_to_episodic_memory(run_id: str) -> dict:
    """Persist a completed run's plan + outcome to the ChromaDB episodic_memory
    collection so EpisodicMemoryTool can retrieve it on future runs.

    This is an explicit user action, so it is not gated by EPISODIC_ENABLED or the
    auto-save quality score.
    """
    from retrieval import Retriever
    from retrieval.chunker import Chunker

    store = SqliteStore()
    run = await store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") != "done":
        raise HTTPException(
            status_code=409,
            detail=f"Run is not complete (status: {run.get('status')})",
        )

    plans = run.get("plans") or []
    plan = plans[0] if plans else None
    plan_id = (plan or {}).get("plan_id", "manual")
    collection = settings.stores.chroma_episodic_collection
    summary = _episodic_summary(run, plan)

    # Write the embedded summary chunk to ChromaDB (retrieval/ owns ChromaDB access).
    def _write_chroma() -> None:
        retriever = Retriever()
        chunker = Chunker(settings.retrieval.chunk_size, settings.retrieval.chunk_overlap)
        docs = chunker.chunk_as_one(summary, source=run_id, domain="episodic")
        retriever.add(collection, docs)

    try:
        await asyncio.to_thread(_write_chroma)
    except Exception as exc:
        logger.exception("episodic save: ChromaDB write failed for run %s", run_id)
        raise HTTPException(status_code=500, detail=f"Episodic save failed: {exc}")

    # Mirror into SQLite (for the Archive browser) and the in-process MCP store.
    entry = {
        "run_id": run_id,
        "plan_id": plan_id,
        "query": run.get("query") or "",
        "exec_summary": run.get("exec_summary") or "",
        "plan": plan,
    }
    try:
        await store.write_figure(run_id, "episodic", plan_id, entry)
        from state_bus.server import record_episodic
        record_episodic(entry)
    except Exception as exc:
        logger.warning("episodic save: secondary persistence failed (non-fatal): %s", exc)

    return {"status": "saved", "run_id": run_id, "collection": collection}


# ── Soft-timeout confirmation ────────────────────────────────────────────────

@router.post("/runs/{run_id}/timeout_confirm/continue", response_model=GateDecisionResponse)
async def continue_after_timeout(
    run_id: str,
    background_tasks: BackgroundTasks,
) -> GateDecisionResponse:
    """User chose to keep the agent running after the soft-timeout prompt."""
    from api.routers.chat import _resume_graph  # delayed import avoids circular dependency

    store = SqliteStore()
    run = await store.get_run(run_id)
    if run is None or run.get("status") != "waiting_timeout_confirm":
        raise HTTPException(
            status_code=404,
            detail="Run not found or not paused at timeout confirmation",
        )
    session_id = run.get("session_id", "")
    query = run.get("query", "")
    await store.upsert_run(run_id, session_id, query, status="running", stage=run.get("stage", ""))
    background_tasks.add_task(_resume_graph, run_id, session_id, query, "approve")
    return GateDecisionResponse(run_id=run_id, gate=0, decision="approve", next_status="running")


@router.post("/runs/{run_id}/timeout_confirm/stop", response_model=GateDecisionResponse)
async def stop_after_timeout(
    run_id: str,
    background_tasks: BackgroundTasks,
) -> GateDecisionResponse:
    """User chose to stop the agent and receive a partial report."""
    from api.routers.chat import _resume_graph  # delayed import avoids circular dependency

    store = SqliteStore()
    run = await store.get_run(run_id)
    if run is None or run.get("status") != "waiting_timeout_confirm":
        raise HTTPException(
            status_code=404,
            detail="Run not found or not paused at timeout confirmation",
        )
    session_id = run.get("session_id", "")
    query = run.get("query", "")
    await store.upsert_run(run_id, session_id, query, status="running", stage=run.get("stage", ""))
    background_tasks.add_task(_resume_graph, run_id, session_id, query, "redirect")
    return GateDecisionResponse(run_id=run_id, gate=0, decision="redirect", next_status="running")


# ── Clarification ───────────────────────────────────────────────────────────

@router.post("/runs/{run_id}/clarify")
async def clarify_run(
    run_id: str,
    body: ClarificationAnswer,
    background_tasks: BackgroundTasks,
) -> dict:
    from api.routers.chat import _run_graph  # avoid circular import at module level

    store = SqliteStore()
    run = await store.get_run(run_id)
    if run is None or run.get("status") != "waiting_clarification":
        raise HTTPException(
            status_code=404,
            detail="Run not found or not waiting for clarification",
        )

    # Persist entity answers to preferences
    for key in ("equipment_models", "mine_sites", "competitor_tickers"):
        val = getattr(body, key)
        if val:
            await store.set_preference(key, val)

    # Reconstruct resumed state
    initial_state = run.get("initial_state") or {}
    session_id = run.get("session_id", "")
    query = run.get("query", "")

    resumed_state = {
        **initial_state,
        "clarification_done": True,
        "stage": "understand",
        "error": None,
    }

    await store.upsert_run(run_id, session_id, query, status="running", stage="understand")
    background_tasks.add_task(_run_graph, run_id, session_id, query, resumed_state)
    return {"status": "resumed", "run_id": run_id}
