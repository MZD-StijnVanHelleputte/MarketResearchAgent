from fastapi import APIRouter
from models.llm_client import LLMClient
from tools.analyst_estimates_tool import AnalystEstimatesTool
from tools.balance_sheet_tool import BalanceSheetTool
from tools.broad_commodity_cycle_tool import BroadCommodityCycleTool
from tools.cash_flow_tool import CashFlowTool
from tools.company_financials_tool import CompanyFinancialsTool
from tools.company_rating_tool import CompanyRatingTool
from tools.earnings_surprises_tool import EarningsSurprisesTool
from tools.energy_cost_prices_tool import EnergyCostPricesTool
from tools.equity_price_tool import EquityPriceTool
from tools.financial_ratios_tool import FinancialRatiosTool
from tools.income_statement_tool import IncomeStatementTool
from tools.industry_knowledge_tool import IndustryKnowledgeTool
from tools.episodic_memory_tool import EpisodicMemoryTool
from tools.macro_indicators_tool import MacroIndicatorsTool
from tools.masterdata_lookup_tool import MasterdataLookupTool
from tools.mining_metals_prices_tool import MiningMetalsPricesTool
from tools.news_search_tool import NewsSearchTool
from tools.press_releases_tool import PressReleasesTool
from tools.sec_filings_tool import SecFilingsTool
from tools.stock_peers_tool import StockPeersTool
from tools.stock_screener_tool import StockScreenerTool
from tools.web_search_tool import WebSearchTool

router = APIRouter(prefix="/tests", tags=["tests"])


@router.post("/run/llm")
async def test_llm():
    """Call the configured LLM with a minimal prompt and verify a response is returned."""
    client = LLMClient()
    try:
        response = await client.acomplete([{"role": "user", "content": "Reply with exactly: OK"}])
        passed = bool(response.content) or len(response.tool_calls) > 0
        output = response.content or f"tool_calls={[tc.name for tc in response.tool_calls]}"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/mining-metals-prices")
async def test_mining_metals_prices():
    """Run get_mining_metals_prices for COPPER and verify observations are returned."""
    tool = MiningMetalsPricesTool()
    try:
        result = await tool.run(symbol="COPPER", interval="monthly")
        latest = result.get("latest") or {}
        value = latest.get("value")
        passed = isinstance(value, (int, float)) and value > 0
        output = f"COPPER: {value} {result.get('unit', '')} (as of {latest.get('date', 'unknown')})"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/energy-cost-prices")
async def test_energy_cost_prices():
    """Run get_energy_cost_prices for WTI and verify observations are returned."""
    tool = EnergyCostPricesTool()
    try:
        result = await tool.run(symbol="WTI", interval="monthly")
        rows = result.get("rows", [])
        latest = result.get("latest") or {}
        passed = len(rows) > 0 and latest.get("value") is not None
        output = f"{len(rows)} row(s) returned for {result.get('symbol')} ({result.get('interval')}); latest: {latest.get('value')}"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/broad-commodity-cycle")
async def test_broad_commodity_cycle():
    """Run get_broad_commodity_cycle and verify broad index observations are returned."""
    tool = BroadCommodityCycleTool()
    try:
        result = await tool.run(interval="monthly")
        rows = result.get("rows", [])
        latest = result.get("latest") or {}
        passed = len(rows) > 0 and latest.get("value") is not None
        output = f"{len(rows)} ALL_COMMODITIES row(s); latest: {latest.get('value')} ({latest.get('date')})"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/company-financials")
async def test_company_financials():
    """Run the get_company_financials tool for CAT (Caterpillar) and verify revenue is returned."""
    tool = CompanyFinancialsTool()
    try:
        result = await tool.run(ticker="CAT")
        passed = result.get("ticker") == "CAT" and result.get("revenue_usd") is not None
        rev = result.get("revenue_usd")
        rev_str = f"${rev/1e9:.1f}B" if rev else "n/a"
        output = f"{result.get('name', 'CAT')} — revenue: {rev_str}, market cap: ${(result.get('market_cap_usd') or 0)/1e9:.1f}B"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/equity-price")
async def test_equity_price():
    """Run the get_equity_price tool for CAT and verify a market price is returned."""
    tool = EquityPriceTool()
    try:
        result = await tool.run(ticker="CAT")
        price = result.get("price")
        passed = price is not None and price > 0
        output = f"CAT: {result.get('currency', 'USD')} {price:.2f} (as of {result.get('date', 'unknown')})"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/sec-filings")
async def test_sec_filings():
    """Run the search_sec_filings tool with a Komatsu query and verify filings are returned."""
    tool = SecFilingsTool()
    try:
        result = await tool.run(query="Komatsu construction equipment")
        filings = result.get("filings", [])
        passed = len(filings) > 0
        output = f"{len(filings)} filing(s) returned"
        if filings:
            f = filings[0]
            output += f'; first: {f.get("entity_name")} {f.get("form_type")} ({f.get("file_date")})'
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/web-search")
async def test_web_search():
    """Run the web_search tool with a Komatsu query and verify results are returned."""
    tool = WebSearchTool()
    try:
        result = await tool.run(query="Komatsu excavator market 2024", max_results=3)
        results = result.get("results", [])
        passed = len(results) > 0
        output = f"{len(results)} result(s) returned"
        if results:
            output += f'; first: "{results[0].get("title", "")}"'
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/masterdata-lookup")
async def test_masterdata_lookup():
    """Run the masterdata_lookup tool for competitors and verify entries are returned."""
    tool = MasterdataLookupTool()
    try:
        result = await tool.run(entity_type="competitors")
        entries = result.get("results", [])
        passed = len(entries) > 0
        output = f"{len(entries)} competitor(s) in master data"
        if entries:
            first = entries[0]
            name = first.get("name") or first.get("ticker") or str(first)
            output += f'; first: {name}'
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/macro-indicator")
async def test_macro_indicator():
    """Run the get_macro_indicator tool for INDPRO and verify observations are returned."""
    tool = MacroIndicatorsTool()
    try:
        result = await tool.run(series_id="INDPRO", limit=5)
        obs = result.get("observations", [])
        passed = len(obs) > 0
        output = f"{result.get('title', 'INDPRO')} ({result.get('units', '')}): {len(obs)} observation(s)"
        if obs:
            latest = obs[0]
            output += f'; latest: {latest.get("date")} = {latest.get("value")}'
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/industry-knowledge")
async def test_industry_knowledge():
    """Search the RAG industry knowledge base and verify the retriever responds without error."""
    tool = IndustryKnowledgeTool()
    try:
        result = await tool.run(query="mining equipment market trends", top_k=3)
        chunks = result.get("results", [])
        passed = True
        output = f"{len(chunks)} chunk(s) retrieved from industry knowledge base"
        if chunks:
            output += f'; top score: {chunks[0].get("score")}, source: {chunks[0].get("source")}'
        else:
            output += " (collection may not be seeded yet)"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/episodic-memory")
async def test_episodic_memory():
    """Search the RAG episodic memory store and verify the retriever responds without error."""
    tool = EpisodicMemoryTool()
    try:
        result = await tool.run(query="Komatsu market analysis", top_k=3)
        chunks = result.get("results", [])
        passed = True
        output = f"{len(chunks)} chunk(s) retrieved from episodic memory"
        if chunks:
            output += f'; top score: {chunks[0].get("score")}, source: {chunks[0].get("source")}'
        else:
            output += " (no prior runs stored yet)"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/news-search")
async def test_news_search():
    """Run the news_search tool with a known query and verify articles are returned."""
    tool = NewsSearchTool()
    try:
        result = await tool.run(query="Komatsu mining equipment", page_size=3)
        articles = result.get("articles", [])
        passed = len(articles) > 0
        output = f"{len(articles)} article(s) returned"
        if articles:
            output += f'; first: "{articles[0]["title"]}"'
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/income-statement")
async def test_income_statement():
    """Run get_income_statement for CAT and verify P&L history is returned."""
    tool = IncomeStatementTool()
    try:
        result = await tool.run(ticker="CAT", period="annual", limit=2)
        statements = result.get("statements", [])
        passed = len(statements) > 0 and statements[0].get("revenue") is not None
        if statements:
            rev = statements[0].get("revenue")
            rev_str = f"${rev/1e9:.1f}B" if rev else "n/a"
            output = f"{len(statements)} period(s) returned; latest revenue: {rev_str} ({statements[0].get('date', '')})"
        else:
            output = "0 periods returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/balance-sheet")
async def test_balance_sheet():
    """Run get_balance_sheet for CAT and verify balance sheet history is returned."""
    tool = BalanceSheetTool()
    try:
        result = await tool.run(ticker="CAT", period="annual", limit=2)
        statements = result.get("statements", [])
        passed = len(statements) > 0 and statements[0].get("total_assets") is not None
        if statements:
            assets = statements[0].get("total_assets")
            assets_str = f"${assets/1e9:.1f}B" if assets else "n/a"
            debt = statements[0].get("total_debt")
            debt_str = f"${debt/1e9:.1f}B" if debt else "n/a"
            output = f"{len(statements)} period(s); total assets: {assets_str}, total debt: {debt_str} ({statements[0].get('date', '')})"
        else:
            output = "0 periods returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/cash-flow")
async def test_cash_flow():
    """Run get_cash_flow for CAT and verify cash flow history is returned."""
    tool = CashFlowTool()
    try:
        result = await tool.run(ticker="CAT", period="annual", limit=2)
        statements = result.get("statements", [])
        passed = len(statements) > 0 and statements[0].get("free_cf") is not None
        if statements:
            fcf = statements[0].get("free_cf")
            fcf_str = f"${fcf/1e9:.1f}B" if fcf else "n/a"
            capex = statements[0].get("capex")
            capex_str = f"${abs(capex)/1e9:.1f}B" if capex else "n/a"
            output = f"{len(statements)} period(s); free cash flow: {fcf_str}, capex: {capex_str} ({statements[0].get('date', '')})"
        else:
            output = "0 periods returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/financial-ratios")
async def test_financial_ratios():
    """Run get_financial_ratios for CAT and verify valuation ratios are returned."""
    tool = FinancialRatiosTool()
    try:
        result = await tool.run(ticker="CAT", period="annual", limit=2)
        ratios = result.get("ratios", [])
        passed = len(ratios) > 0 and ratios[0].get("pe_ratio") is not None
        if ratios:
            pe = ratios[0].get("pe_ratio")
            ev_ebitda = ratios[0].get("ev_ebitda")
            output = f"{len(ratios)} period(s); P/E: {pe:.1f}, EV/EBITDA: {ev_ebitda:.1f if ev_ebitda else 'n/a'} ({ratios[0].get('date', '')})"
        else:
            output = "0 periods returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/analyst-estimates")
async def test_analyst_estimates():
    """Run get_analyst_estimates for CAT and verify consensus estimates are returned."""
    tool = AnalystEstimatesTool()
    try:
        result = await tool.run(ticker="CAT", period="annual", limit=2)
        estimates = result.get("estimates", [])
        passed = len(estimates) > 0 and estimates[0].get("est_revenue_avg") is not None
        if estimates:
            rev = estimates[0].get("est_revenue_avg")
            rev_str = f"${rev/1e9:.1f}B" if rev else "n/a"
            analysts = estimates[0].get("num_analysts")
            output = f"{len(estimates)} period(s); est. revenue: {rev_str}, analyst count: {analysts} ({estimates[0].get('date', '')})"
        else:
            output = "0 periods returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/stock-peers")
async def test_stock_peers():
    """Run get_stock_peers for CAT and verify a peer list is returned."""
    tool = StockPeersTool()
    try:
        result = await tool.run(ticker="CAT")
        peers = result.get("peers", [])
        passed = len(peers) > 0
        output = f"{len(peers)} peer(s) identified: {', '.join(peers[:5])}" if peers else "0 peers returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/company-rating")
async def test_company_rating():
    """Run get_company_rating for CAT and verify a DCF-based rating is returned."""
    tool = CompanyRatingTool()
    try:
        result = await tool.run(ticker="CAT")
        ratings = result.get("ratings", [])
        passed = len(ratings) > 0 and ratings[0].get("rating") is not None
        if ratings:
            output = f"Rating: {ratings[0].get('rating')}, score: {ratings[0].get('score')} (as of {ratings[0].get('date', 'unknown')})"
        else:
            output = "0 ratings returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/earnings-surprises")
async def test_earnings_surprises():
    """Run get_earnings_surprises for CAT and verify beat/miss history is returned."""
    tool = EarningsSurprisesTool()
    try:
        result = await tool.run(ticker="CAT", limit=4)
        surprises = result.get("surprises", [])
        passed = len(surprises) > 0
        if surprises:
            s = surprises[0]
            pct = s.get("surprise_pct")
            pct_str = f"{pct:+.1f}%" if pct is not None else "n/a"
            output = f"{len(surprises)} quarter(s); latest: actual EPS {s.get('actual_eps')}, est. {s.get('estimated_eps')}, surprise {pct_str} ({s.get('date', '')})"
        else:
            output = "0 quarters returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/press-releases")
async def test_press_releases():
    """Run get_press_releases for CAT and verify official press releases are returned."""
    tool = PressReleasesTool()
    try:
        result = await tool.run(ticker="CAT", limit=3)
        releases = result.get("press_releases", [])
        passed = len(releases) > 0
        output = f"{len(releases)} press release(s) returned"
        if releases:
            output += f'; latest: "{releases[0].get("title", "")}" ({releases[0].get("date", "")})'
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}


@router.post("/run/stock-screener")
async def test_stock_screener():
    """Run screen_stocks for Industrials sector in the US and verify results are returned."""
    tool = StockScreenerTool()
    try:
        result = await tool.run(sector="Industrials", country="US", limit=5)
        results = result.get("results", [])
        passed = len(results) > 0
        if results:
            names = [r.get("ticker", "") for r in results[:3]]
            output = f"{len(results)} company(ies) matched; sample: {', '.join(names)}"
        else:
            output = "0 results returned"
    except Exception as exc:
        return {"passed": False, "output": str(exc)}
    return {"passed": passed, "output": output}
