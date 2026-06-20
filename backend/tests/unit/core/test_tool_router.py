"""Unit tests for core/tool_router.py — specifically the new stage_tools() function."""
import pytest

from core.tool_router import stage_tools
from tools.registry import COLLECT_TOOLS, UNDERSTAND_TOOLS, SYNTHESIZE_TOOLS


def test_stage_tools_returns_collect_list():
    tools = stage_tools("collect")
    assert len(tools) == len(COLLECT_TOOLS)
    names = {t.name for t in tools}
    assert "news_search" in names
    assert "get_mining_metals_prices" in names
    assert "get_energy_cost_prices" in names
    assert "get_broad_commodity_cycle" in names
    assert "web_search" in names


def test_stage_tools_returns_understand_list():
    tools = stage_tools("understand")
    names = {t.name for t in tools}
    assert "search_industry_knowledge" in names
    assert "search_episodic_memory" in names


def test_stage_tools_returns_synthesize_list():
    tools = stage_tools("synthesize")
    names = {t.name for t in tools}
    assert "search_industry_knowledge" in names


def test_stage_tools_unknown_stage_returns_empty():
    assert stage_tools("unknown_stage") == []


def test_stage_tools_filters_by_domain_and_plan():
    plan = {
        "plan_id": "p1",
        "tool_calls": [
            {"tool": "get_company_financials", "domain": "competition", "arguments": {}},
            {"tool": "get_equity_price",        "domain": "competition", "arguments": {}},
            {"tool": "get_mining_metals_prices", "domain": "commodities", "arguments": {}},
        ],
    }
    competition_tools = stage_tools("collect", domain="competition", plan=plan)
    names = {t.name for t in competition_tools}
    assert "get_company_financials" in names
    assert "get_equity_price" in names
    # commodity tool should NOT be in competition filter
    assert "get_mining_metals_prices" not in names


def test_stage_tools_domain_with_no_matching_tool_calls():
    plan = {
        "plan_id": "p1",
        "tool_calls": [
            {"tool": "get_mining_metals_prices", "domain": "commodities", "arguments": {}},
        ],
    }
    tools = stage_tools("collect", domain="competition", plan=plan)
    assert tools == []


def test_stage_tools_domain_none_returns_full_list():
    tools = stage_tools("collect", domain=None, plan=None)
    assert len(tools) == len(COLLECT_TOOLS)
