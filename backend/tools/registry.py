from config import settings as _settings
from core.domains import domain_keys, domain_tools
from tools.agricultural_commodity_prices_tool import AgriculturalCommodityPricesTool
from tools.analyst_estimates_tool import AnalystEstimatesTool
from tools.balance_sheet_tool import BalanceSheetTool
from tools.base import BaseTool
from tools.broad_commodity_cycle_tool import BroadCommodityCycleTool
from tools.cash_flow_tool import CashFlowTool
from tools.company_financials_tool import CompanyFinancialsTool
from tools.company_rating_tool import CompanyRatingTool
from tools.earnings_calendar_tool import EarningsCalendarTool
from tools.earnings_surprises_tool import EarningsSurprisesTool
from tools.earnings_transcript_tool import EarningsTranscriptTool
from tools.energy_cost_prices_tool import EnergyCostPricesTool
from tools.equity_financials_tool import EquityFinancialsTool
from tools.equity_history_tool import EquityHistoryTool
from tools.equity_price_tool import EquityPriceTool
from tools.episodic_memory_tool import EpisodicMemoryTool
from tools.financial_ratios_tool import FinancialRatiosTool
from tools.fx_rates_tool import FxRatesTool
from tools.income_statement_tool import IncomeStatementTool
from tools.industry_knowledge_tool import IndustryKnowledgeTool
from tools.insider_transactions_tool import InsiderTransactionsTool
from tools.fred_category_tool import FredCategoryTool
from tools.fred_observations_tool import FredObservationsTool
from tools.fred_release_series_tool import FredReleaseSeriesToolCls
from tools.fred_releases_tool import FredReleasesTool
from tools.fred_search_tool import FredSearchTool
from tools.fred_series_updates_tool import FredSeriesUpdatesTool
from tools.fred_tags_series_tool import FredTagsSeriesTool
from tools.macro_indicators_tool import MacroIndicatorsTool
from tools.masterdata_lookup_tool import MasterdataLookupTool
from tools.mining_metals_prices_tool import MiningMetalsPricesTool
from tools.news_search_tool import NewsSearchTool
from tools.news_sentiment_tool import NewsSentimentTool
from tools.news_sources_tool import NewsSourcesTool
from tools.news_top_headlines_tool import NewsTopHeadlinesTool
from tools.press_releases_tool import PressReleasesTool
from tools.sec_filings_tool import SecFilingsTool
from tools.stock_peers_tool import StockPeersTool
from tools.stock_screener_tool import StockScreenerTool
from tools.technical_report_tool import TechnicalReportTool
from tools.web_crawl_tool import WebCrawlTool
from tools.web_extract_tool import WebExtractTool
from tools.web_map_tool import WebMapTool
from tools.web_research_tool import WebResearchTool
from tools.web_search_tool import WebSearchTool

# Shared tool instances (same object in multiple stage lists)
_episodic = EpisodicMemoryTool()
_knowledge = IndustryKnowledgeTool()
_web_search = WebSearchTool()
_news_search = NewsSearchTool()
_news_top_headlines = NewsTopHeadlinesTool()
_news_sources = NewsSourcesTool()
_web_extract = WebExtractTool()
_masterdata = MasterdataLookupTool()

# Stage-based tool lists — the tool_router selects the correct list per stage.
# The same instance may appear in multiple lists.
UNDERSTAND_TOOLS: list[BaseTool] = [
    _knowledge,
    _episodic,
    _web_search,      # live competitor / entity discovery
    _news_search,     # recent headlines to frame plans
    _news_top_headlines,  # breaking news to frame plans
    _web_extract,     # follow a specific URL (e.g., investor page)
    _masterdata,      # resolve entities against master data first
]
COLLECT_TOOLS: list[BaseTool] = [
    _news_search,
    _news_top_headlines,
    _news_sources,
    MiningMetalsPricesTool(),
    EnergyCostPricesTool(),
    BroadCommodityCycleTool(),
    AgriculturalCommodityPricesTool(),
    FxRatesTool(),
    CompanyFinancialsTool(),
    SecFilingsTool(),
    EquityPriceTool(),
    EquityHistoryTool(),
    EquityFinancialsTool(),
    EarningsCalendarTool(),
    NewsSentimentTool(),
    EarningsTranscriptTool(),
    InsiderTransactionsTool(),
    _web_search,
    _web_extract,
    WebCrawlTool(),
    WebMapTool(),
    WebResearchTool(),
    _masterdata,
    MacroIndicatorsTool(),
    FredSearchTool(),
    FredObservationsTool(),
    FredReleasesTool(),
    FredReleaseSeriesToolCls(),
    FredCategoryTool(),
    FredTagsSeriesTool(),
    FredSeriesUpdatesTool(),
    IncomeStatementTool(),
    BalanceSheetTool(),
    CashFlowTool(),
    FinancialRatiosTool(),
    AnalystEstimatesTool(),
    StockPeersTool(),
    CompanyRatingTool(),
    EarningsSurprisesTool(),
    PressReleasesTool(),
    StockScreenerTool(),
    TechnicalReportTool(),
]
SYNTHESIZE_TOOLS: list[BaseTool] = [_knowledge, _episodic]


def _tier_filter(tools: list[BaseTool]) -> list[BaseTool]:
    """Drop tools that require a premium subscription when the matching tier is 'free'."""
    keep = []
    for t in tools:
        req = getattr(t, "requires_premium", None)
        if req == "fmp" and _settings.fmp_tier == "free":
            continue
        if req == "alpha_vantage" and _settings.alpha_vantage_tier == "free":
            continue
        keep.append(t)
    return keep


COLLECT_TOOLS = _tier_filter(COLLECT_TOOLS)

# Flat name → tool lookup (built from all stage lists)
_registry: dict[str, BaseTool] = {
    t.name: t
    for stage_list in (UNDERSTAND_TOOLS, COLLECT_TOOLS, SYNTHESIZE_TOOLS)
    for t in stage_list
}


def get(name: str) -> BaseTool:
    if name not in _registry:
        raise KeyError(f"Tool '{name}' is not registered.")
    return _registry[name]


def all_tools() -> list[BaseTool]:
    return list(_registry.values())


# Internal tool name → friendly display name, shared by the API (sources panel)
# and the domain agents (Gate 2 / failed-tool labels) so both speak one vocabulary.
TOOL_DISPLAY_NAMES: dict[str, str] = {
    "news_search": "NewsAPI",
    "web_search": "WebSearch",
    "search_sec_filings": "SEC EDGAR",
    "get_mining_metals_prices": "Mining Metals (Alpha Vantage)",
    "get_energy_cost_prices": "Energy Costs (Alpha Vantage)",
    "get_broad_commodity_cycle": "Commodity Cycle (Alpha Vantage)",
    "get_company_financials": "Financials (FMP)",
    "get_equity_price": "Equity Prices",
    "get_macro_indicator": "Macro (FRED)",
    "masterdata_lookup": "Master Data",
}


def tool_display_name(tool: str) -> str:
    """Friendly label for a tool, falling back to the raw name."""
    return TOOL_DISPLAY_NAMES.get(tool, tool or "unknown")


# Domain → tool name mapping (authoritative reference for domain sub-agents).
# Derived from the domain registry: each domain's tools are the union of the
# toolsets of every leaf type it may hold (core/domains.LEAF_TOOLSETS). The
# actual per-run filter uses plan["tool_calls"][].domain via stage_tools().
DOMAIN_TOOLS: dict[str, list[str]] = {
    domain: domain_tools(domain) for domain in domain_keys()
}
