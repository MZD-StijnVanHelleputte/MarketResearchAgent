"""FastMCP planning state bus.

Exposes plan state and episodic memory via the MCP protocol so that any
MCP-capable client (including Claude desktop, VS Code, etc.) can inspect
the current run's candidate plans.

In-process callers should use write_plan_direct() / get_all_plans_direct()
to avoid MCP protocol overhead.

Note: this module lives in state_bus/ (not mcp/) to avoid shadowing the
'mcp' pip package that FastMCP depends on.
"""
import logging
from typing import Any

from fastmcp import FastMCP

from config import settings

logger = logging.getLogger(__name__)

mcp_server = FastMCP("Komatsu Planning State Bus")

# ---------------------------------------------------------------------------
# In-memory store (keyed by plan_id; also persisted to SqliteStore when available)
# ---------------------------------------------------------------------------
_plans: dict[str, dict] = {}
_episodic: list[dict] = []


# ---------------------------------------------------------------------------
# MCP resources
# ---------------------------------------------------------------------------

@mcp_server.resource("planning://plans")
def get_plans() -> dict:
    """All CandidatePlan objects (both depths) for the current run."""
    return {"plans": list(_plans.values())}


@mcp_server.resource("planning://coverage")
def get_coverage() -> dict:
    """Domain coverage checklist across all survivor plans."""
    from collections import Counter
    domain_counts: Counter = Counter()
    for plan in _plans.values():
        if plan.get("is_survivor"):
            for domain, active in plan.get("domain_activations", {}).items():
                if active:
                    domain_counts[domain] += 1
    return {"coverage": dict(domain_counts)}


@mcp_server.resource("episodic://past_plans")
def get_past_plans() -> dict:
    """Past successful execution plans stored in episodic memory."""
    if not settings.stores.episodic_enabled:
        return {"past_plans": [], "enabled": False}
    return {"past_plans": _episodic, "enabled": True}


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@mcp_server.tool()
def write_plan(plan: dict) -> dict:
    """Upsert a CandidatePlan into the state bus."""
    plan_id = plan.get("plan_id")
    if not plan_id:
        return {"status": "error", "reason": "plan_id is required"}
    _plans[plan_id] = plan
    logger.debug("state_bus: upserted plan %s (depth=%s)", plan_id, plan.get("depth"))
    return {"status": "ok", "plan_id": plan_id}


@mcp_server.tool()
def write_episodic(run_id: str, plan_id: str, outcome: str) -> dict:
    """Append a completed run's plan + outcome to episodic memory."""
    if not settings.stores.episodic_enabled:
        return {"status": "skipped", "reason": "episodic memory is disabled"}

    plan_data = _plans.get(plan_id)
    if plan_data is None:
        return {"status": "error", "reason": f"plan_id '{plan_id}' not found"}

    entry = {"run_id": run_id, "plan_id": plan_id, "outcome": outcome, "plan": plan_data}
    _episodic.append(entry)

    try:
        from memory.sqlite_store import SqliteStore
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            SqliteStore().write_figure(run_id, "episodic", plan_id, entry)
        )
    except Exception as exc:
        logger.warning("state_bus: sqlite persistence failed: %s", exc)

    return {"status": "ok", "run_id": run_id, "plan_id": plan_id}


# ---------------------------------------------------------------------------
# Direct in-process API (no MCP protocol overhead)
# ---------------------------------------------------------------------------

def write_plan_direct(plan: Any) -> None:
    """Write a CandidatePlan to the state bus directly (in-process)."""
    data = plan.model_dump() if hasattr(plan, "model_dump") else dict(plan)
    _plans[data["plan_id"]] = data


def get_all_plans_direct() -> list[dict]:
    """Return all plans currently in the state bus."""
    return list(_plans.values())


def clear_plans() -> None:
    """Clear all plans (call at the start of each new chat run)."""
    _plans.clear()


def record_episodic(entry: dict) -> None:
    """Append a finished run's episodic entry to the in-process MCP store.

    Used by the manual 'save to episodic memory' endpoint; unlike write_episodic()
    this is not gated by episodic_enabled (the user explicitly opted in) and does
    not perform its own persistence (the caller writes ChromaDB + SQLite directly).
    """
    _episodic.append(entry)
