import pytest
from unittest.mock import patch, AsyncMock
from core.plan_merger import PlanMerger, _dedupe_cross_domain, _expand_per_ticker_calls, _fallback_merge
from core.tot.schemas import CandidatePlan, PlannedToolCall, ResearchContext


def _candidate(plan_id="plan_001", tool_calls=None, domains=None):
    return CandidatePlan(
        plan_id=plan_id,
        domain_activations=domains or {"competition": True},
        tool_calls=tool_calls or [],
        feasibility_score=0.8,
        quality_score=0.7,
        combined_score=0.75,
        rationale="test rationale",
    )


def test_expand_per_ticker_calls_adds_missing_tickers():
    planned = [
        PlannedToolCall(tool="get_equity_history", params={"ticker": "CAT", "period": "5y"}, domain="competition"),
    ]
    entity_manifest = {"tickers": ["CAT", "DE", "SAND.ST"]}

    expanded = _expand_per_ticker_calls(planned, entity_manifest)

    tickers_called = {tc.params["ticker"] for tc in expanded if tc.tool == "get_equity_history"}
    assert tickers_called == {"CAT", "DE", "SAND.ST"}
    # Non-ticker params (e.g. period) are preserved from the template
    for tc in expanded:
        if tc.tool == "get_equity_history":
            assert tc.params["period"] == "5y"


def test_expand_per_ticker_calls_does_not_duplicate_existing():
    planned = [
        PlannedToolCall(tool="get_equity_price", params={"ticker": "CAT"}, domain="competition"),
        PlannedToolCall(tool="get_equity_price", params={"ticker": "DE"}, domain="competition"),
    ]
    entity_manifest = {"tickers": ["CAT", "DE"]}

    expanded = _expand_per_ticker_calls(planned, entity_manifest)

    assert len(expanded) == 2


def test_expand_per_ticker_calls_ignores_non_ticker_tools():
    planned = [
        PlannedToolCall(tool="get_mining_metals_prices", params={"symbol": "COPPER"}, domain="commodities"),
    ]
    entity_manifest = {"tickers": ["CAT", "DE"]}

    expanded = _expand_per_ticker_calls(planned, entity_manifest)

    assert expanded == planned


def test_expand_per_ticker_calls_no_tickers_is_noop():
    planned = [
        PlannedToolCall(tool="get_equity_price", params={"ticker": "CAT"}, domain="competition"),
    ]
    expanded = _expand_per_ticker_calls(planned, {"tickers": []})
    assert expanded == planned


def test_expand_per_ticker_calls_respects_safety_cap():
    planned = [
        PlannedToolCall(tool="get_equity_price", params={"ticker": "CAT"}, domain="competition"),
    ]
    many_tickers = [f"T{i}" for i in range(200)]
    entity_manifest = {"tickers": many_tickers}

    with patch("core.plan_merger.settings") as mock_settings:
        mock_settings.safety.max_api_calls_per_run = 50
        expanded = _expand_per_ticker_calls(planned, entity_manifest)

    assert len(expanded) == 50


def test_dedupe_cross_domain_keeps_higher_priority_domain():
    calls = [
        PlannedToolCall(tool="get_mining_metals_prices", params={"symbol": "COPPER"}, domain="macro_geopolitics"),
        PlannedToolCall(tool="get_mining_metals_prices", params={"symbol": "COPPER"}, domain="commodities"),
    ]

    deduped = _dedupe_cross_domain(calls)

    assert len(deduped) == 1
    assert deduped[0].domain == "commodities"


def test_dedupe_cross_domain_keeps_distinct_params():
    calls = [
        PlannedToolCall(tool="get_mining_metals_prices", params={"symbol": "COPPER"}, domain="commodities"),
        PlannedToolCall(tool="get_mining_metals_prices", params={"symbol": "GOLD"}, domain="commodities"),
    ]

    deduped = _dedupe_cross_domain(calls)

    assert len(deduped) == 2


def test_fallback_merge_dedupes_across_domains():
    survivors = [
        _candidate(plan_id="plan_001", tool_calls=[
            {"tool": "get_mining_metals_prices", "domain": "commodities", "arguments": {"symbol": "COPPER"}},
        ]),
        _candidate(plan_id="plan_002", tool_calls=[
            {"tool": "get_mining_metals_prices", "domain": "macro_geopolitics", "arguments": {"symbol": "COPPER"}},
        ]),
    ]
    research_context = ResearchContext()

    plan = _fallback_merge(survivors, research_context, run_id="run123")

    copper_calls = [tc for tc in plan.planned_tool_calls if tc.tool == "get_mining_metals_prices"]
    assert len(copper_calls) == 1
    assert copper_calls[0].domain == "commodities"


def test_fallback_merge_expands_per_ticker():
    survivors = [
        _candidate(tool_calls=[
            {"tool": "get_equity_history", "domain": "competition", "arguments": {"ticker": "CAT", "period": "5y"}},
        ]),
    ]
    research_context = ResearchContext(tickers=["CAT", "DE", "VOLV-B.ST"])

    plan = _fallback_merge(survivors, research_context, run_id="run123")

    tickers_called = {
        tc.params["ticker"] for tc in plan.planned_tool_calls if tc.tool == "get_equity_history"
    }
    assert tickers_called == {"CAT", "DE", "VOLV-B.ST"}


@pytest.mark.asyncio
async def test_merge_llm_path_expands_per_ticker():
    survivors = [
        _candidate(tool_calls=[
            {"tool": "get_equity_price", "domain": "competition", "arguments": {"ticker": "CAT"}},
        ]),
    ]
    research_context = ResearchContext(tickers=["CAT", "DE"])

    merger = PlanMerger()
    llm_response_json = (
        '{"plan_id": "x", "domains_active": ["competition"], "entity_manifest": {}, '
        '"planned_tool_calls": [{"tool": "get_equity_price", "domain": "competition", '
        '"params": {"ticker": "CAT"}}], "research_findings": "", "rationale": "r", '
        '"gap_report": "", "feasibility_score": 0.8, "quality_score": 0.7}'
    )

    class _FakeResponse:
        content = llm_response_json
        usage = {}

    with patch.object(merger._llm, "acomplete", AsyncMock(return_value=_FakeResponse())):
        plan = await merger.merge(survivors, research_context, run_id="run123")

    tickers_called = {
        tc.params["ticker"] for tc in plan.planned_tool_calls if tc.tool == "get_equity_price"
    }
    assert tickers_called == {"CAT", "DE"}


@pytest.mark.asyncio
async def test_merge_llm_path_dedupes_cross_domain_duplicates():
    survivors = [
        _candidate(tool_calls=[
            {"tool": "get_mining_metals_prices", "domain": "commodities", "arguments": {"symbol": "COPPER"}},
        ]),
    ]
    research_context = ResearchContext()

    merger = PlanMerger()
    llm_response_json = (
        '{"plan_id": "x", "domains_active": ["commodities", "macro_geopolitics"], '
        '"entity_manifest": {}, "planned_tool_calls": ['
        '{"tool": "get_mining_metals_prices", "domain": "commodities", "params": {"symbol": "COPPER"}}, '
        '{"tool": "get_mining_metals_prices", "domain": "macro_geopolitics", "params": {"symbol": "COPPER"}}'
        '], "research_findings": "", "rationale": "r", "gap_report": "", '
        '"feasibility_score": 0.8, "quality_score": 0.7}'
    )

    class _FakeResponse:
        content = llm_response_json
        usage = {}

    with patch.object(merger._llm, "acomplete", AsyncMock(return_value=_FakeResponse())):
        plan = await merger.merge(survivors, research_context, run_id="run123")

    copper_calls = [tc for tc in plan.planned_tool_calls if tc.tool == "get_mining_metals_prices"]
    assert len(copper_calls) == 1
    assert copper_calls[0].domain == "commodities"
