"""Run-scoped tool circuit breaker.

Tracks tool failures at the *tool* level (not per logical call), so a tool that
keeps failing — for any company, ticker, or time window — is blocked for the rest
of the run instead of being re-hammered on every collect→backtrack retry and the
recovery pass. A tool is blocked after `tool_failure_threshold` logical-call
failures, or immediately on the first rate-limit (HTTP 429): a throttled API key
won't recover mid-run, so retrying it just wastes minutes and pollutes synthesis
with empty data.

State is in-memory and keyed by run_id. Tool calls within a run share one asyncio
event loop, so plain dict mutation is safe without locking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import settings


@dataclass
class _ToolState:
    fail_count: int = 0
    blocked: bool = False
    reason: str = ""


# run_id -> tool_name -> _ToolState
_runs: dict[str, dict[str, _ToolState]] = {}


def reset_run(run_id: str) -> None:
    """Clear all breaker state for a run (call at run start)."""
    _runs.pop(run_id, None)


def _tools(run_id: str) -> dict[str, _ToolState]:
    return _runs.setdefault(run_id, {})


def is_blocked(run_id: str | None, tool_name: str) -> bool:
    if not run_id:
        return False
    state = _runs.get(run_id, {}).get(tool_name)
    return bool(state and state.blocked)


def block_reason(run_id: str | None, tool_name: str) -> str:
    if not run_id:
        return ""
    state = _runs.get(run_id, {}).get(tool_name)
    return state.reason if state else ""


def record_success(run_id: str | None, tool_name: str) -> None:
    """Note a successful logical call. A success does not un-block an already
    blocked tool (the block is sticky for the rest of the run) but does ensure the
    tool has a state entry for the health summary."""
    if not run_id:
        return
    _tools(run_id).setdefault(tool_name, _ToolState())


def record_failure(run_id: str | None, tool_name: str, exc: BaseException | None) -> bool:
    """Record one logical-call failure. Returns True if this failure just blocked
    the tool (so the caller can emit a one-time `tool_blocked` event)."""
    if not run_id:
        return False
    state = _tools(run_id).setdefault(tool_name, _ToolState())
    if state.blocked:
        return False
    state.fail_count += 1
    if exc is not None and is_rate_limited_error(exc):
        state.blocked = True
        state.reason = "rate limited (HTTP 429) — blocked for the rest of the run"
        return True
    if state.fail_count >= settings.react.tool_failure_threshold:
        state.blocked = True
        state.reason = (
            f"failed {state.fail_count} times — blocked for the rest of the run"
        )
        return True
    return False


def summary(run_id: str | None) -> dict[str, dict]:
    """Per-tool failure counts and block reasons, for the end-of-collect health event."""
    if not run_id:
        return {}
    return {
        tool: {"fail_count": s.fail_count, "blocked": s.blocked, "reason": s.reason}
        for tool, s in _runs.get(run_id, {}).items()
        if s.fail_count or s.blocked
    }


_RATE_LIMIT_MARKERS = (
    "rate limit",
    "ratelimit",
    "429",
    "too many requests",
    # Alpha Vantage returns HTTP 200 with this note instead of a 429 when throttled.
    "our standard api rate limit",
    "thank you for using alpha vantage",
)


def is_rate_limited_error(exc: BaseException) -> bool:
    """True if a failure (or any cause in its chain) is a rate-limit signal — an
    HTTP 429, or a provider message that means 'throttled'. Walks __cause__/__context__
    like is_permanent_tool_error so wrapped exceptions are caught too."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if getattr(cur, "status", None) == 429:
            return True
        text = str(cur).lower()
        if any(marker in text for marker in _RATE_LIMIT_MARKERS):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class ToolBlockedError(RuntimeError):
    """Raised by the tool router when a tool is short-circuited by the circuit breaker.
    Treated as a permanent error so callers fail fast without LLM argument repair."""

    def __init__(self, tool_name: str, reason: str = "") -> None:
        self.tool_name = tool_name
        self.reason = reason
        super().__init__(f"Tool '{tool_name}' is blocked: {reason or 'too many failures'}")
