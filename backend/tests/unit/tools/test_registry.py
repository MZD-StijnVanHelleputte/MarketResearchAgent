import pytest
from tools import registry
from tools.news_search_tool import NewsSearchTool
from tools.news_top_headlines_tool import NewsTopHeadlinesTool
from tools.news_sources_tool import NewsSourcesTool
from tools.equity_financials_tool import EquityFinancialsTool


def test_news_search_is_collect_tool():
    assert any(isinstance(t, NewsSearchTool) for t in registry.COLLECT_TOOLS)


def test_news_top_headlines_is_collect_and_understand_tool():
    assert any(isinstance(t, NewsTopHeadlinesTool) for t in registry.COLLECT_TOOLS)
    assert any(isinstance(t, NewsTopHeadlinesTool) for t in registry.UNDERSTAND_TOOLS)


def test_news_sources_is_collect_tool_only():
    assert any(isinstance(t, NewsSourcesTool) for t in registry.COLLECT_TOOLS)
    assert not any(isinstance(t, NewsSourcesTool) for t in registry.UNDERSTAND_TOOLS)
    assert not any(isinstance(t, NewsSourcesTool) for t in registry.SYNTHESIZE_TOOLS)


def test_equity_financials_is_collect_tool():
    assert any(isinstance(t, EquityFinancialsTool) for t in registry.COLLECT_TOOLS)


def test_get_equity_financials_by_name():
    tool = registry.get("get_equity_financials")
    assert tool.name == "get_equity_financials"


def test_domain_tools_reference_equity_financials():
    for domain in ("competition", "mining_operators", "construction_companies", "specialized_customers"):
        assert "get_equity_financials" in registry.DOMAIN_TOOLS[domain]


def test_get_news_top_headlines_by_name():
    tool = registry.get("news_top_headlines")
    assert tool.name == "news_top_headlines"


def test_get_news_sources_by_name():
    tool = registry.get("news_sources")
    assert tool.name == "news_sources"


def test_all_tools_includes_new_news_tools():
    names = [t.name for t in registry.all_tools()]
    assert "news_top_headlines" in names
    assert "news_sources" in names


def test_domain_tools_reference_new_news_tools():
    for domain in ("competition", "distributors", "mining_operators",
                   "construction_companies", "specialized_customers", "general_search"):
        assert "news_top_headlines" in registry.DOMAIN_TOOLS[domain]
    assert "news_sources" in registry.DOMAIN_TOOLS["general_search"]


def test_news_search_in_understand_not_synthesize():
    # news_search frames plans with recent headlines during Understand, but is not
    # part of the Synthesize stage (which only reads knowledge + episodic memory).
    assert any(isinstance(t, NewsSearchTool) for t in registry.UNDERSTAND_TOOLS)
    assert not any(isinstance(t, NewsSearchTool) for t in registry.SYNTHESIZE_TOOLS)


def test_get_news_search_by_name():
    tool = registry.get("news_search")
    assert tool.name == "news_search"


def test_get_unknown_tool_raises():
    with pytest.raises(KeyError, match="not registered"):
        registry.get("nonexistent_tool")


def test_all_tools_includes_news_search():
    names = [t.name for t in registry.all_tools()]
    assert "news_search" in names
