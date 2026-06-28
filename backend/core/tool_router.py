"""Routes LLM tool-call requests to registered tools, enforcing TOOL_ALLOWLIST."""
import time

from config import settings
from clients.base_http_client import PERMANENT_STATUSES
from core.event_logger import get_run_context, log_event, record_live_source
from core import tool_circuit_breaker as breaker
from core.tool_circuit_breaker import ToolBlockedError
import tools.registry as registry


def is_permanent_tool_error(exc: BaseException) -> bool:
    """True if a tool failure (or any cause in its chain) is a permanent HTTP client
    error — bad request / not found / unprocessable. Such errors can never succeed on
    retry or via LLM argument-repair (e.g. FRED returns HTTP 400 'The series does not
    exist' for an invalid series_id), so callers should fail fast instead of looping.
    A breaker block (ToolBlockedError) is permanent too — the tool is dead for the run."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, ToolBlockedError):
            return True
        status = getattr(cur, "status", None)
        if isinstance(status, int) and status in PERMANENT_STATUSES:
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def route(tool_name: str, tool_input: dict) -> dict:
    if settings.safety.tool_allowlist and tool_name not in settings.safety.tool_allowlist:
        raise PermissionError(f"Tool '{tool_name}' is not in TOOL_ALLOWLIST.")
    return registry.get(tool_name).run(tool_input)


async def async_route(
    tool_name: str,
    tool_input: dict,
    call_id: str | None = None,
    count_failures: bool = True,
) -> dict:
    if settings.safety.tool_allowlist and tool_name not in settings.safety.tool_allowlist:
        raise PermissionError(f"Tool '{tool_name}' is not in TOOL_ALLOWLIST.")
    ctx = get_run_context()
    run_id = ctx.get("run_id")
    domain_tag = f" [{ctx['domain']}]" if ctx.get("domain") else ""
    # Circuit breaker: a tool blocked earlier this run is skipped without an API call,
    # so it can't be re-hammered on backtrack/recovery or pollute synthesis.
    if breaker.is_blocked(run_id, tool_name):
        reason = breaker.block_reason(run_id, tool_name)
        await log_event(
            "tool_blocked", f"{tool_name}{domain_tag} skipped (circuit breaker)",
            detail={"args": tool_input, "reason": reason, "call_id": call_id},
            level="warning",
        )
        raise ToolBlockedError(tool_name, reason)
    t0 = time.monotonic()
    try:
        result = await registry.get(tool_name).run(**tool_input)
        latency_ms = int((time.monotonic() - t0) * 1000)
        first_key = next(iter(result), "ok") if isinstance(result, dict) else "ok"
        await log_event(
            "tool_call", f"{tool_name}{domain_tag}",
            detail={
                "args": tool_input, "result_key": first_key, "latency_ms": latency_ms,
                "call_id": call_id,
            },
        )
        if count_failures:
            breaker.record_success(run_id, tool_name)
        # Grow the Sources panel one source at a time as each tool returns.
        await record_live_source(tool_name, tool_input, result=result)
        return result
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        await log_event(
            "tool_error", f"{tool_name}{domain_tag} FAILED",
            detail={
                "args": tool_input, "error": str(exc), "latency_ms": latency_ms,
                "call_id": call_id,
            },
            level="error",
        )
        # Direct callers (research/recovery) count failures here; the domain-agent repair
        # loop passes count_failures=False and records one logical outcome itself, so a
        # single call's 5 repair sub-attempts don't trip the 5-failure threshold.
        if count_failures and not isinstance(exc, ToolBlockedError):
            if breaker.record_failure(run_id, tool_name, exc):
                await log_event(
                    "tool_blocked", f"{tool_name}{domain_tag} now blocked (circuit breaker)",
                    detail={"reason": breaker.block_reason(run_id, tool_name), "call_id": call_id},
                    level="warning",
                )
        await record_live_source(tool_name, tool_input, failed=True, reason=str(exc))
        raise


def stage_tools(stage: str, domain: str | None = None, plan: dict | None = None) -> list:
    """Return tool instances for a pipeline stage, optionally filtered by domain+plan.

    If both domain and plan are provided, returns only tools whose names appear
    in that plan's tool_calls tagged with the given domain. Used by domain
    sub-agents to discover which tools are assigned to them in a specific plan.
    """
    stage_map = {
        "understand": registry.UNDERSTAND_TOOLS,
        "collect": registry.COLLECT_TOOLS,
        "synthesize": registry.SYNTHESIZE_TOOLS,
    }
    tools = list(stage_map.get(stage, []))
    if domain is None or plan is None:
        return tools
    # Support both ConsolidatedPlan ("planned_tool_calls") and CandidatePlan ("tool_calls")
    raw_calls = plan.get("planned_tool_calls") or plan.get("tool_calls") or []
    assigned_names = {tc.get("tool") for tc in raw_calls if tc.get("domain") == domain}
    return [t for t in tools if t.name in assigned_names]
