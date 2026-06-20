"""Business-friendly display names for tools and domains, used in the
live activity feed surfaced to the chat UI. Internal identifiers (tool
function names, domain keys) should never reach the frontend directly.
"""

TOOL_DISPLAY_NAMES: dict[str, str] = {
    "news_search": "news articles",
    "news_top_headlines": "top headlines",
    "news_sources": "news sources",
    "get_news_sentiment": "news sentiment",
    "web_search": "the web",
    "web_extract": "a source page",
    "web_crawl": "a source site",
    "web_map": "a source site's pages",
    "web_research": "in-depth web research",
    "search_sec_filings": "SEC filings",
    "search_industry_knowledge": "our industry knowledge base",
    "search_episodic_memory": "past research",
    "masterdata_lookup": "our master data",
    "get_mining_metals_prices": "mining & metals prices",
    "get_agricultural_commodity_prices": "agricultural commodity prices",
    "get_energy_cost_prices": "energy cost data",
    "get_broad_commodity_cycle": "commodity cycle data",
    "get_macro_indicator": "macroeconomic indicators",
    "get_fx_rates": "currency exchange rates",
    "get_company_financials": "company financials",
    "get_equity_price": "equity prices",
    "get_equity_history": "equity price history",
    "get_equity_financials": "equity financials",
    "get_balance_sheet": "balance sheet data",
    "get_cash_flow": "cash flow data",
    "get_income_statement": "income statement data",
    "get_financial_ratios": "financial ratios",
    "get_analyst_estimates": "analyst estimates",
    "get_earnings_calendar": "the earnings calendar",
    "get_earnings_surprises": "earnings surprises",
    "get_earnings_transcript": "earnings call transcripts",
    "get_insider_transactions": "insider transactions",
    "get_press_releases": "press releases",
    "get_company_rating": "company ratings",
    "get_stock_peers": "peer company data",
    "screen_stocks": "stock screening data",
    "search_fred_series": "economic data series",
    "browse_fred_category": "economic data categories",
    "get_fred_observations": "economic indicator data",
    "get_fred_release_series": "economic data releases",
    "list_fred_releases": "economic data releases",
    "get_fred_series_by_tags": "tagged economic data series",
    "get_fred_series_updates": "updated economic data series",
}

DOMAIN_DISPLAY_NAMES: dict[str, str] = {
    "mining_projects": "Mining Projects",
    "distributors": "Distributors",
    "commodities": "Commodities",
    "macro_geopolitics": "Macro & Geopolitics",
    "customers": "Customers",
    "general_search": "General Market Search",
    "competition": "Competition",
}


def friendly_tool(tool_name: str) -> str:
    return TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " "))


def friendly_domain(domain: str) -> str:
    return DOMAIN_DISPLAY_NAMES.get(domain, domain.replace("_", " ").title())
