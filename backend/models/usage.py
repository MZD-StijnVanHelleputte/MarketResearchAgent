"""Token-usage accounting for the live cost counter.

A "usage" dict is the common currency passed around the graph:
    {"prompt_tokens": int, "completion_tokens": int, "requests": int}

`requests` counts billable calls (LLM calls + external tool/API calls) and feeds
the run's api_call_count. Cost is derived from token counts using the configurable
per-1M-token prices in settings.llm. There are no spend caps — this is tracking only.
"""
from config import settings


def usd_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """USD cost for a given token split, using configured per-1M-token prices."""
    p = settings.llm.input_price_per_1m
    c = settings.llm.output_price_per_1m
    return round((prompt_tokens * p + completion_tokens * c) / 1_000_000, 6)


def llm_usage(usage: dict | None) -> dict:
    """Normalise an LLMResponse.usage dict into the common usage shape (1 request)."""
    usage = usage or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "requests": 1 if usage else 0,
    }


def crew_usage(crew_output) -> dict:
    """Extract token usage from a CrewAI CrewOutput.token_usage (UsageMetrics)."""
    tu = getattr(crew_output, "token_usage", None)
    if tu is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}
    return {
        "prompt_tokens": int(getattr(tu, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(tu, "completion_tokens", 0) or 0),
        "requests": int(getattr(tu, "successful_requests", 0) or 0),
    }


def merge_usage(*usages: dict) -> dict:
    """Sum any number of usage dicts into one."""
    pt = ct = rq = 0
    for u in usages:
        if not u:
            continue
        pt += int(u.get("prompt_tokens", 0) or 0)
        ct += int(u.get("completion_tokens", 0) or 0)
        rq += int(u.get("requests", 0) or 0)
    return {"prompt_tokens": pt, "completion_tokens": ct, "requests": rq}


def accumulate(state: dict, *usages: dict) -> dict:
    """Return a state-delta with running token/cost/call totals updated.

    Reads the prior totals from *state* and adds the given usage dicts, so a node
    that runs more than once (e.g. a backtrack re-running collect) keeps adding.
    """
    delta = merge_usage(*usages)
    pt = int(state.get("prompt_tokens", 0) or 0) + delta["prompt_tokens"]
    ct = int(state.get("completion_tokens", 0) or 0) + delta["completion_tokens"]
    calls = int(state.get("api_call_count", 0) or 0) + delta["requests"]
    return {
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": pt + ct,
        "api_call_count": calls,
        "cumulative_cost_usd": usd_cost(pt, ct),
    }
