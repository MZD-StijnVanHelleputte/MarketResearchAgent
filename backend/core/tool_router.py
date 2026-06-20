"""Routes LLM tool-call requests to registered tools, enforcing TOOL_ALLOWLIST."""
import time

from config import settings
from core.event_logger import get_run_context, log_event
import tools.registry as registry


def route(tool_name: str, tool_input: dict) -> dict:
    if settings.safety.tool_allowlist and tool_name not in settings.safety.tool_allowlist:
        raise PermissionError(f"Tool '{tool_name}' is not in TOOL_ALLOWLIST.")
    return registry.get(tool_name).run(tool_input)


async def async_route(tool_name: str, tool_input: dict) -> dict:
    if settings.safety.tool_allowlist and tool_name not in settings.safety.tool_allowlist:
        raise PermissionError(f"Tool '{tool_name}' is not in TOOL_ALLOWLIST.")
    ctx = get_run_context()
    domain_tag = f" [{ctx['domain']}]" if ctx.get("domain") else ""
    t0 = time.monotonic()
    try:
        result = await registry.get(tool_name).run(**tool_input)
        latency_ms = int((time.monotonic() - t0) * 1000)
        first_key = next(iter(result), "ok") if isinstance(result, dict) else "ok"
        await log_event(
            "tool_call", f"{tool_name}{domain_tag}",
            detail={"args": tool_input, "result_key": first_key, "latency_ms": latency_ms},
        )
        return result
    except Exception as exc:
        latency_ms = int((time.monotonic() - t0) * 1000)
        await log_event(
            "tool_error", f"{tool_name}{domain_tag} FAILED",
            detail={"args": tool_input, "error": str(exc), "latency_ms": latency_ms},
            level="error",
        )
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
