"""Lightweight per-step event logger.

Uses a ContextVar so run_id + domain propagate through async call stacks
without being passed explicitly to every helper.
"""
import asyncio
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

_run_ctx: ContextVar[dict] = ContextVar("run_ctx", default={})
logger = logging.getLogger(__name__)

# One lock per run guards read-modify-write of the runs.sources column, which is
# updated concurrently by the per-tool live writer here and by graph's per-domain
# _persist_partial_sources. Both acquire THIS lock so neither loses an update.
_run_locks: dict[str, asyncio.Lock] = {}


def run_lock(run_id: str) -> asyncio.Lock:
    lock = _run_locks.get(run_id)
    if lock is None:
        lock = _run_locks[run_id] = asyncio.Lock()
    return lock


def set_run_context(run_id: str, stage: str = "", domain: str = "") -> object:
    return _run_ctx.set({"run_id": run_id, "stage": stage, "domain": domain})


def update_run_context(**kwargs) -> None:
    ctx = dict(_run_ctx.get())
    ctx.update(kwargs)
    _run_ctx.set(ctx)


def get_run_context() -> dict:
    return _run_ctx.get()


async def log_event(
    event_type: str,
    label: str,
    detail: dict | None = None,
    level: str = "info",
    stage: str | None = None,
    domain: str | None = None,
) -> None:
    ctx = _run_ctx.get()
    run_id = ctx.get("run_id")
    if not run_id:
        return

    from memory.sqlite_store import SqliteStore
    ts = datetime.now(timezone.utc).isoformat()
    try:
        await SqliteStore().log_step_event(
            run_id=run_id,
            ts=ts,
            level=level,
            stage=stage if stage is not None else ctx.get("stage", ""),
            domain=domain if domain is not None else ctx.get("domain", ""),
            event_type=event_type,
            label=label,
            detail=detail,
        )
    except Exception as exc:
        logger.debug("event_logger: failed to persist event: %s", exc)


_LABEL_ARG_KEYS = ("ticker", "symbol", "series_id", "mine_name", "company", "query", "q", "keywords")


def _provisional_label(tool_input: dict) -> str:
    """Best-effort short label for a tool call (e.g. a ticker or query term)."""
    if not isinstance(tool_input, dict):
        return ""
    for key in _LABEL_ARG_KEYS:
        val = tool_input.get(key)
        if val:
            return str(val)[:40]
    return ""


def _provisional_count(result: object) -> int:
    """Best-effort row/item count from a raw tool result."""
    if isinstance(result, dict):
        for val in result.values():
            if isinstance(val, list):
                return len(val)
    return 0


async def record_live_source(
    tool_name: str,
    tool_input: dict,
    result: object | None = None,
    failed: bool = False,
    reason: str = "",
) -> None:
    """Append one provisional source row to runs.sources the moment a tool returns,
    so the frontend Sources panel grows source-by-source during collection instead of
    in per-domain chunks. Entries are marked provisional and replaced by graph's typed
    per-domain entries (and the authoritative Gate-2 rebuild) as the run progresses.

    Best-effort and self-contained: never raises into the tool path.
    """
    ctx = _run_ctx.get()
    run_id = ctx.get("run_id")
    if not run_id:
        return
    # Only the collect stage feeds the Sources panel; research/understand tool calls
    # provide planning context, not panel sources, and would leave orphan provisional
    # rows (the per-domain cleanup in graph runs only during collect).
    if (ctx.get("stage") or "") != "collect":
        return
    domain = ctx.get("domain", "")
    label = _provisional_label(tool_input)
    try:
        from memory.sqlite_store import SqliteStore
        from tools.registry import tool_display_name

        display = tool_display_name(tool_name)
        entry = {
            "domain": domain,
            "tool": display,
            "title": display,
            "data_type": "failed" if failed else "data",
            "label": label,
            "count": 0 if failed else _provisional_count(result),
            "url": None,
            "published_at": None,
            "failed": failed,
            "provisional": True,
        }
        if failed and reason:
            entry["reason"] = reason

        store = SqliteStore()
        async with run_lock(run_id):
            row = await store.get_run(run_id)
            if row is None:
                return
            existing = list(row.get("sources") or [])
            # Dedup: skip if an equivalent row (same domain/tool/label/failed) is present.
            key = (domain, display, label, failed)
            if any(
                (s.get("domain"), s.get("tool"), s.get("label"), s.get("failed")) == key
                for s in existing
            ):
                return
            existing.append(entry)
            await store.upsert_run(
                run_id, row.get("session_id", ""), row.get("query"),
                status="running", stage=ctx.get("stage") or "collect",
                sources=existing,
            )
    except Exception as exc:
        logger.debug("event_logger: failed to record live source: %s", exc)
