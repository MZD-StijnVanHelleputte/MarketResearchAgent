"""Lightweight per-step event logger.

Uses a ContextVar so run_id + domain propagate through async call stacks
without being passed explicitly to every helper.
"""
import logging
from contextvars import ContextVar
from datetime import datetime, timezone

_run_ctx: ContextVar[dict] = ContextVar("run_ctx", default={})
logger = logging.getLogger(__name__)


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
