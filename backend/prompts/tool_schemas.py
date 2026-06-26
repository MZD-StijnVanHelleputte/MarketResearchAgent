"""Mistral function-calling schemas surfaced to the LLM for tool calling.
Each entry mirrors the Pydantic input_schema of the corresponding tool."""

from config import settings as _settings

NEWS_SEARCH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "news_search",
        "description": (
            "Search recent news articles by keyword. "
            "Returns titles, descriptions, URLs, and publication dates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or phrase.",
                },
                "language": {
                    "type": "string",
                    "description": "Two-letter ISO 639-1 language code (default 'en').",
                    "default": "en",
                },
                "page_size": {
                    "type": "integer",
                    "description": "Number of articles to return (1–20, default 5).",
                    "default": 5,
                },
                "from_date": {
                    "type": "string",
                    "description": (
                        "Earliest article date in YYYY-MM-DD format. "
                        "Defaults to 30 days ago if omitted. "
                        "For trend analysis use 90 days; for breaking news use 7 days."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

NEWS_TOP_HEADLINES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "news_top_headlines",
        "description": (
            "Get breaking/top news headlines, optionally filtered by country, category, "
            "or specific source IDs (use news_sources to discover valid source IDs). "
            "'sources' cannot be combined with 'country' or 'category'. "
            "Use for breaking-news monitoring; use news_search for full-text historical search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional keyword filter on headline text.",
                },
                "country": {
                    "type": "string",
                    "description": (
                        "Two-letter country code, e.g. 'us', 'au', 'za', 'cl'. "
                        "Cannot combine with 'sources'."
                    ),
                },
                "category": {
                    "type": "string",
                    "description": (
                        "One of: business, entertainment, general, health, science, sports, "
                        "technology. Cannot combine with 'sources'."
                    ),
                    "enum": [
                        "business",
                        "entertainment",
                        "general",
                        "health",
                        "science",
                        "sports",
                        "technology",
                    ],
                },
                "sources": {
                    "type": "string",
                    "description": (
                        "Comma-separated NewsAPI source IDs (from news_sources). "
                        "Cannot combine with 'country' or 'category'."
                    ),
                },
                "page_size": {
                    "type": "integer",
                    "description": "Number of headlines to return (1–20, default 5).",
                    "default": 5,
                },
                "page": {
                    "type": "integer",
                    "description": "Page number for pagination (default 1).",
                    "default": 1,
                },
            },
            "required": [],
        },
    },
}

NEWS_SOURCES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "news_sources",
        "description": (
            "List available NewsAPI news sources/publishers, optionally filtered by "
            "category, language, or country. Returns source id, name, description, "
            "url, category, language, country. Use the returned 'id' values as the "
            "'sources' parameter for news_top_headlines or news_search."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": (
                        "One of: business, entertainment, general, health, science, sports, technology."
                    ),
                    "enum": [
                        "business",
                        "entertainment",
                        "general",
                        "health",
                        "science",
                        "sports",
                        "technology",
                    ],
                },
                "language": {
                    "type": "string",
                    "description": "Two-letter ISO 639-1 language code, e.g. 'en'.",
                },
                "country": {
                    "type": "string",
                    "description": "Two-letter country code, e.g. 'us', 'au'.",
                },
            },
            "required": [],
        },
    },
}

MINING_METALS_PRICES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_mining_metals_prices",
        "description": (
            "Get Alpha Vantage mining metal price data for COPPER, ALUMINUM, GOLD, "
            "or SILVER. Use for mining demand cycles and metal-linked equipment demand. "
            "Returns symbol, endpoint, interval, unit, latest observation, rows, and source."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "enum": ["COPPER", "ALUMINUM", "GOLD", "SILVER", "XAU", "XAG"],
                    "description": (
                        "Commodity code — never a company/equity stock ticker. "
                        "One of COPPER, ALUMINUM, GOLD, SILVER, XAU, or XAG."
                    ),
                },
                "interval": {
                    "type": "string",
                    "description": (
                        "For GOLD/SILVER: daily, weekly, monthly. "
                        "For COPPER/ALUMINUM: monthly, quarterly, annual."
                    ),
                    "default": "monthly",
                },
                "include_history": {
                    "type": "boolean",
                    "description": "For GOLD/SILVER, false uses the live GOLD_SILVER_SPOT endpoint.",
                    "default": True,
                },
            },
            "required": ["symbol"],
        },
    },
}

ENERGY_COST_PRICES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_energy_cost_prices",
        "description": (
            "Get Alpha Vantage WTI, BRENT, or NATURAL_GAS price series. "
            "Use for fuel, energy-cost, and project economics signals."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "One of WTI, BRENT, or NATURAL_GAS.",
                },
                "interval": {
                    "type": "string",
                    "description": "daily, weekly, or monthly.",
                    "default": "monthly",
                },
            },
            "required": ["symbol"],
        },
    },
}

BROAD_COMMODITY_CYCLE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_broad_commodity_cycle",
        "description": (
            "Get Alpha Vantage ALL_COMMODITIES broad commodity index data. "
            "Use for broad construction/mining cycle context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "interval": {
                    "type": "string",
                    "description": "monthly, quarterly, or annual.",
                    "default": "monthly",
                },
            },
            "required": [],
        },
    },
}

COMPANY_FINANCIALS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_company_financials",
        "description": (
            "Get the latest financial summary for a publicly traded company by ticker symbol. "
            "Returns revenue, net income, capex, market cap, and P/E ratio."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'SAND.ST', '6305.T'.",
                },
            },
            "required": ["ticker"],
        },
    },
}

SEC_FILINGS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_sec_filings",
        "description": (
            "Search SEC EDGAR filings by keyword. Returns matching filing metadata "
            "including entity name, form type, filing date, and reporting period."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords (company name, topic, etc.).",
                },
                "forms": {
                    "type": "string",
                    "description": "Comma-separated form types to filter by (default '10-K,10-Q,8-K').",
                    "default": "10-K,10-Q,8-K",
                },
            },
            "required": ["query"],
        },
    },
}

TECHNICAL_REPORT_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_mine_technical_report",
        "description": (
            "Fetch the SEC S-K 1300 Technical Report Summary (Exhibit 96 of a mining "
            "company's most recent 10-K/20-F) — mineral resource/reserve estimates, mine "
            "life, and project economics for one company. Requires the company's stock "
            "ticker; use mine_name to pick one project when the filer reports several."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker of the mining company, e.g. 'FCX', 'BHP'.",
                },
                "mine_name": {
                    "type": "string",
                    "description": "Optional mine/project name to disambiguate when the "
                    "filer's 10-K includes multiple Exhibit 96 reports, e.g. 'Morenci'.",
                },
            },
            "required": ["ticker"],
        },
    },
}

EQUITY_PRICE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_equity_price",
        "description": (
            "Get the latest market price, currency, and market cap for a stock ticker. "
            "Works for major exchanges (NYSE, NASDAQ, TSE, LSE, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker, e.g. 'CAT', 'DE', 'SAND.ST'.",
                },
            },
            "required": ["ticker"],
        },
    },
}

WEB_SEARCH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the open web via Tavily. Returns titles, URLs, and content snippets. "
            "Use search_depth='advanced' for precision, topic='news' for recent events, "
            "topic='finance' for financial data, time_range to restrict recency, "
            "include_domains to focus on trusted sources (e.g. 'sec.gov,reuters.com')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords or question (under 400 characters).",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results (1–20, default 5).",
                    "default": 5,
                },
                "search_depth": {
                    "type": "string",
                    "description": (
                        "Precision level: 'basic' (default, fast), 'advanced' (slower, higher "
                        "precision), 'fast' (good with chunks), 'ultra-fast' (real-time)."
                    ),
                    "default": "basic",
                },
                "topic": {
                    "type": "string",
                    "description": "'general' (default), 'news' (recent events), 'finance' (financial data).",
                    "default": "general",
                },
                "time_range": {
                    "type": "string",
                    "description": "Restrict results to: 'day', 'week', 'month', or 'year'. Optional.",
                },
                "include_domains": {
                    "type": "string",
                    "description": "Comma-separated domain allowlist, e.g. 'sec.gov,reuters.com'. Optional.",
                    "default": "",
                },
                "exclude_domains": {
                    "type": "string",
                    "description": "Comma-separated domain blocklist. Optional.",
                    "default": "",
                },
                "include_answer": {
                    "type": "boolean",
                    "description": "When true, prepend an AI-generated answer summary.",
                    "default": False,
                },
                "include_raw_content": {
                    "type": "boolean",
                    "description": "When true, return full page text alongside snippets.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}

WEB_EXTRACT_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_extract",
        "description": (
            "Extract clean full-page content from specific URLs via Tavily /extract. "
            "Use after web_search returns URLs to read in full: competitor press releases, "
            "mining project pages, IR sections, regulatory publications. "
            "Returns a list of pages with url and content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "urls": {
                    "type": "string",
                    "description": "Comma-separated URLs to extract (up to 20).",
                },
                "query": {
                    "type": "string",
                    "description": "Optional relevance query to rerank extracted chunks.",
                    "default": "",
                },
                "extract_depth": {
                    "type": "string",
                    "description": "'basic' (default) or 'advanced' (for JS-rendered pages).",
                    "default": "basic",
                },
            },
            "required": ["urls"],
        },
    },
}

WEB_CRAWL_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_crawl",
        "description": (
            "Crawl a website and extract content from multiple pages via Tavily /crawl. "
            "Use to bulk-extract a competitor's IR section, mine operator's project pages, "
            "or regulatory publications. Set instructions for semantic focus to avoid context overflow. "
            "Use web_map first to understand site structure. Returns pages_crawled and page list."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Root URL to start crawling from.",
                },
                "instructions": {
                    "type": "string",
                    "description": (
                        "Natural language semantic focus, e.g. 'Find product announcements about "
                        "mining equipment'. When set, returns relevant chunks only."
                    ),
                    "default": "",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Link-levels deep to crawl (1–5, default 1).",
                    "default": 1,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max pages to crawl (1–50, default 20). Keep low.",
                    "default": 20,
                },
                "select_paths": {
                    "type": "string",
                    "description": "Comma-separated regex path filters, e.g. '/news/.*,/press/.*'. Optional.",
                    "default": "",
                },
            },
            "required": ["url"],
        },
    },
}

WEB_MAP_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_map",
        "description": (
            "Discover URLs on a website without extracting content via Tavily /map. "
            "Use before web_crawl or web_extract to understand site structure and find target pages. "
            "Typical workflow: web_map competitor IR section → identify URLs → web_extract specific pages. "
            "Returns root_url and list of discovered URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Root URL of the site or section to map.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Natural language URL filter, e.g. 'Find mining equipment pages'. Optional.",
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max URLs to discover (1–200, default 50).",
                    "default": 50,
                },
            },
            "required": ["url"],
        },
    },
}

WEB_RESEARCH_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "web_research",
        "description": (
            "AI-synthesized multi-source research with citations via Tavily /research. "
            "Returns a structured report — NOT just snippets. Use for deep strategic topics: "
            "competitive landscape, market outlooks, geopolitical risk assessments. "
            "Takes 30–120 seconds. Use model='pro' for complex comparisons, 'mini' for targeted topics. "
            "Returns report text and citation URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Research question, e.g. 'autonomous mining trucks competitive landscape 2025'.",
                },
                "model": {
                    "type": "string",
                    "description": (
                        "'auto' (default, API chooses), 'mini' (~30 s, single-topic), "
                        "'pro' (~60–120 s, comprehensive multi-angle analysis)."
                    ),
                    "default": "auto",
                },
            },
            "required": ["query"],
        },
    },
}

MASTERDATA_LOOKUP_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "masterdata_lookup",
        "description": (
            "Query Komatsu's internal master-data registry. "
            "entity_type: distributors | competitors | operators | equipment | commodities. "
            "Filter by region (e.g. 'Asia-Pacific') or keyword (name, country, ticker, product)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entity_type": {
                    "type": "string",
                    "description": "One of: distributors, competitors, operators, equipment, commodities.",
                    "enum": ["distributors", "competitors", "operators", "equipment", "commodities"],
                },
                "region": {
                    "type": "string",
                    "description": "Optional region filter, e.g. 'Asia-Pacific', 'Americas'.",
                    "default": "",
                },
                "keyword": {
                    "type": "string",
                    "description": "Optional substring match on name, country, ticker, or product.",
                    "default": "",
                },
            },
            "required": ["entity_type"],
        },
    },
}

MACRO_INDICATORS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_macro_indicator",
        "description": (
            "Fetch a macroeconomic time series from FRED (Federal Reserve Economic Data). "
            "Useful series_ids: FEDFUNDS (Fed funds rate), GDP (US GDP), CPIAUCSL (CPI inflation), "
            "INDPRO (industrial production), DGS10 (10Y Treasury rate), HOUST (housing starts), "
            "DCOILWTICO (crude oil WTI), DEXJPUS (JPY/USD), DEXCHUS (CNY/USD), UNRATE (unemployment)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "series_id": {
                    "type": "string",
                    "description": "FRED series ID, e.g. 'FEDFUNDS', 'GDP', 'CPIAUCSL'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of most-recent observations to return (1–50, default 10).",
                    "default": 10,
                },
            },
            "required": ["series_id"],
        },
    },
}

AGRICULTURAL_COMMODITY_PRICES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_agricultural_commodity_prices",
        "description": (
            "Get Alpha Vantage agricultural commodity price series for WHEAT or CORN. "
            "Use as a food-inflation and social-risk signal for emerging-market mining regions "
            "(Africa, South America). Supports monthly, quarterly, and annual intervals."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "One of WHEAT or CORN.",
                    "enum": ["WHEAT", "CORN"],
                },
                "interval": {
                    "type": "string",
                    "description": "monthly, quarterly, or annual.",
                    "default": "monthly",
                },
            },
            "required": ["symbol"],
        },
    },
}

FX_RATES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_fx_rates",
        "description": (
            "Get Alpha Vantage FX exchange-rate series (daily, weekly, or monthly). "
            "Key mining-country pairs: USD/AUD, USD/BRL, USD/CLP, USD/ZAR, USD/JPY. "
            "Use for currency-risk and equipment-pricing analysis. "
            "Note: FRED already covers JPY/USD and CNY/USD via get_macro_indicator."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_currency": {
                    "type": "string",
                    "description": "ISO 4217 base currency code, e.g. 'USD'.",
                },
                "to_currency": {
                    "type": "string",
                    "description": "ISO 4217 quote currency code, e.g. 'AUD'.",
                },
                "interval": {
                    "type": "string",
                    "description": "daily, weekly, or monthly.",
                    "default": "monthly",
                },
            },
            "required": ["from_currency", "to_currency"],
        },
    },
}

EARNINGS_CALENDAR_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_earnings_calendar",
        "description": (
            "Get upcoming earnings announcement dates (Alpha Vantage EARNINGS_CALENDAR). "
            "Use to know when CAT, Volvo CE, Sandvik, Epiroc, or Komatsu will report, "
            "so the competition agent can time deeper analysis. Free tier."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Ticker to filter by, e.g. 'CAT'. Leave empty for market-wide.",
                    "default": "",
                },
                "horizon": {
                    "type": "string",
                    "description": "Lookahead window: 3month, 6month, or 12month.",
                    "default": "3month",
                },
            },
            "required": [],
        },
    },
}

NEWS_SENTIMENT_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_news_sentiment",
        "description": (
            "Get sentiment-scored news articles (Alpha Vantage NEWS_SENTIMENT, premium). "
            "Returns overall sentiment score/label and per-ticker sentiment for each article. "
            "Filter by tickers (e.g. 'CAT,EPIR.ST') or topics (e.g. 'mining,earnings'). "
            "Requires Alpha Vantage premium subscription."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tickers": {
                    "type": "string",
                    "description": "Comma-separated ticker symbols, e.g. 'CAT,VOLV-B.ST'. Optional.",
                    "default": "",
                },
                "topics": {
                    "type": "string",
                    "description": (
                        "Comma-separated topic filters, e.g. 'mining,earnings'. "
                        "Options: earnings, mergers_and_acquisitions, economy_macro, "
                        "energy_transportation, manufacturing."
                    ),
                    "default": "",
                },
                "sort": {
                    "type": "string",
                    "description": "Sort order: LATEST, RELEVANCE, or SENTIMENT.",
                    "default": "LATEST",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of articles to return (1–50).",
                    "default": 25,
                },
            },
            "required": [],
        },
    },
}

EARNINGS_TRANSCRIPT_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_earnings_transcript",
        "description": (
            "Retrieve a parsed earnings call transcript (Alpha Vantage EARNINGS_CALL_TRANSCRIPT, premium). "
            "Use for the competition agent to extract what executives at CAT, Deere, or other "
            "competitors said about equipment demand, mining markets, or product pipeline. "
            "Requires Alpha Vantage premium subscription."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker, e.g. 'CAT', 'DE'. US-listed tickers only.",
                },
                "quarter": {
                    "type": "string",
                    "description": "Fiscal quarter in YYYYQn format, e.g. '2024Q4'.",
                },
            },
            "required": ["symbol", "quarter"],
        },
    },
}

INSIDER_TRANSACTIONS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_insider_transactions",
        "description": (
            "Get recent insider buying/selling activity (Alpha Vantage INSIDER_TRANSACTIONS, premium). "
            "Use for the competition agent (CAT, Deere) or mining_projects agent (FCX, VALE, RIO) "
            "to detect unusual insider activity signalling strategic moves or earnings surprises. "
            "Requires Alpha Vantage premium subscription."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "US-listed stock ticker, e.g. 'CAT', 'FCX', 'VALE'.",
                },
            },
            "required": ["symbol"],
        },
    },
}

INCOME_STATEMENT_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_income_statement",
        "description": (
            "Get historical income statements (P&L) for a publicly traded company. "
            "Returns revenue, gross profit, operating income, net income, EBITDA, and EPS "
            "for up to 20 annual or quarterly periods."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', '6305.T', 'VOLVb.ST'.",
                },
                "period": {
                    "type": "string",
                    "description": "'annual' (default) or 'quarter'. Use 'quarter' for recent quarterly trend.",
                    "default": "annual",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of periods to return (1–20, default 8).",
                    "default": 8,
                },
            },
            "required": ["ticker"],
        },
    },
}

BALANCE_SHEET_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_balance_sheet",
        "description": (
            "Get historical balance sheets for a publicly traded company. "
            "Returns cash, total assets, total debt, total equity, and net debt "
            "for up to 20 annual or quarterly periods."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'DE', 'KOMATSU.T'.",
                },
                "period": {
                    "type": "string",
                    "description": "'annual' (default) or 'quarter'. Use 'quarter' for recent quarterly trend.",
                    "default": "annual",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of periods to return (1–20, default 8).",
                    "default": 8,
                },
            },
            "required": ["ticker"],
        },
    },
}

CASH_FLOW_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_cash_flow",
        "description": (
            "Get historical cash flow statements for a publicly traded company. "
            "Returns operating cash flow, capital expenditure (capex), and free cash flow "
            "for up to 20 annual or quarterly periods. Use for equipment investment cycle signals."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'FCX', 'RIO'.",
                },
                "period": {
                    "type": "string",
                    "description": "'annual' (default) or 'quarter'. Use 'quarter' for recent quarterly trend.",
                    "default": "annual",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of periods to return (1–20, default 8).",
                    "default": 8,
                },
            },
            "required": ["ticker"],
        },
    },
}

FINANCIAL_RATIOS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_financial_ratios",
        "description": (
            "Get historical financial ratios for a publicly traded company. "
            "Returns P/E ratio, EV/EBITDA, debt-to-equity, ROE, ROIC, and current ratio "
            "for up to 20 annual or quarterly periods. Use for valuation benchmarking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'DE', 'HIK.L'.",
                },
                "period": {
                    "type": "string",
                    "description": "'annual' (default) or 'quarter'. Use 'quarter' for recent quarterly trend.",
                    "default": "annual",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of periods to return (1–20, default 8).",
                    "default": 8,
                },
            },
            "required": ["ticker"],
        },
    },
}

ANALYST_ESTIMATES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_analyst_estimates",
        "description": (
            "Get Wall Street analyst consensus estimates for a publicly traded company. "
            "Returns forward revenue and EPS estimates (high, low, average) and analyst count. "
            "Use for forward-looking competitor or customer outlook."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'BHP', 'RIO'.",
                },
                "period": {
                    "type": "string",
                    "description": "'annual' (default) or 'quarter'.",
                    "default": "annual",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of periods to return (1–20, default 8).",
                    "default": 8,
                },
            },
            "required": ["ticker"],
        },
    },
}

STOCK_PEERS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_stock_peers",
        "description": (
            "Get the peer companies for a publicly traded company by ticker symbol. "
            "Returns a list of peer ticker symbols in the same sector and market cap range. "
            "Use to discover competitors for benchmarking."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'DE', 'KOMATSU.T'.",
                },
            },
            "required": ["ticker"],
        },
    },
}

COMPANY_RATING_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_company_rating",
        "description": (
            "Get the overall DCF-based investment rating for a publicly traded company. "
            "Returns a rating label (Strong Buy / Buy / Neutral / Sell / Strong Sell), "
            "an overall score, and sub-scores for DCF, ROE, and debt. "
            "Use for a quick valuation health check on competitors or customers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'DE', 'VOLCAR-B.ST'.",
                },
            },
            "required": ["ticker"],
        },
    },
}

EARNINGS_SURPRISES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_earnings_surprises",
        "description": (
            "Get the earnings surprise history for a publicly traded company. "
            "Returns actual EPS vs analyst consensus estimate and the surprise percentage "
            "for the most recent quarterly reports. Use to assess management execution quality."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'DE', 'BHP'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of quarterly periods to return (1–40, default 8).",
                    "default": 8,
                },
            },
            "required": ["ticker"],
        },
    },
}

PRESS_RELEASES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_press_releases",
        "description": (
            "Get recent official press releases for a publicly traded company. "
            "Returns titles, dates, and text content (up to 2000 chars each). "
            "Use for M&A announcements, strategic partnerships, product launches, and guidance updates."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'HIK.L', 'VOLCAR-B.ST'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of press releases to return (1–50, default 10).",
                    "default": 10,
                },
            },
            "required": ["ticker"],
        },
    },
}

EQUITY_HISTORY_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_equity_history",
        "description": (
            "Get daily OHLCV (open/high/low/close/volume) price history for a stock ticker "
            "over a configurable period (default 1y, up to 5y). Use to analyse year-over-year "
            "price performance, volatility, drawdowns, and multi-year trends for competitors "
            "or major mining operators. Works for major exchanges (NYSE, NASDAQ, TSE, LSE, etc.)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'VOLV-B.ST', '6305.T'.",
                },
                "period": {
                    "type": "string",
                    "description": (
                        "Lookback period for daily OHLCV data. "
                        "Options: 1mo, 3mo, 6mo, 1y, 2y, 5y. Default is 1y."
                    ),
                    "default": "1y",
                },
            },
            "required": ["ticker"],
        },
    },
}

EQUITY_FINANCIALS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_equity_financials",
        "description": (
            "Get multi-year (annual) or multi-quarter income-statement line items "
            "(revenue, net income, operating income, etc.) for a stock ticker via Yahoo "
            "Finance. Free, no premium tier required — use this for long-term financial "
            "trend analysis of competitors when FMP fundamentals tools are unavailable."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'CAT', 'VOLV-B.ST', '6305.T'.",
                },
                "period": {
                    "type": "string",
                    "enum": ["annual", "quarterly"],
                    "description": (
                        "Statement granularity. 'annual' returns ~4 fiscal years; "
                        "'quarterly' returns ~4 most recent quarters. Default is annual."
                    ),
                    "default": "annual",
                },
            },
            "required": ["ticker"],
        },
    },
}

STOCK_SCREENER_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "screen_stocks",
        "description": (
            "Screen for publicly traded companies matching specific criteria. "
            "Filter by sector, industry, country, market cap range, or exchange. "
            "Returns ticker, company name, price, market cap, sector, industry, country, and exchange. "
            "Use to discover unknown competitors, customers, or mining project developers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": (
                        "Sector filter, e.g. 'Industrials', 'Basic Materials', 'Energy', "
                        "'Technology', 'Financial Services'."
                    ),
                },
                "industry": {
                    "type": "string",
                    "description": "Industry filter, e.g. 'Farm & Heavy Construction Machinery'.",
                },
                "country": {
                    "type": "string",
                    "description": "Two-letter country code, e.g. 'US', 'JP', 'SE', 'AU', 'CA'.",
                },
                "market_cap_min": {
                    "type": "number",
                    "description": "Minimum market capitalisation in USD.",
                },
                "market_cap_max": {
                    "type": "number",
                    "description": "Maximum market capitalisation in USD.",
                },
                "exchange": {
                    "type": "string",
                    "description": "Exchange code, e.g. 'NYSE', 'NASDAQ', 'TSX', 'EURONEXT'.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1–100, default 20).",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
}

FRED_SEARCH_SERIES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_fred_series",
        "description": (
            "Search the full FRED (Federal Reserve Economic Data) catalogue by keyword to discover "
            "relevant economic series when you don't know the exact series_id. "
            "Returns matching series with id, title, units, frequency, and popularity. "
            "Example searches: 'steel production', 'construction spending', 'manufacturing PMI', "
            "'crude oil imports', 'China trade', 'copper price', 'truck sales'. "
            "Use the returned series_id with get_fred_observations or get_macro_indicator to fetch data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "search_text": {
                    "type": "string",
                    "description": "Keyword(s) to search for in FRED series titles and descriptions.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of results to return (1–100, default 10).",
                    "default": 10,
                },
            },
            "required": ["search_text"],
        },
    },
}

FRED_OBSERVATIONS_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_fred_observations",
        "description": (
            "Fetch FRED economic time-series observations with full control over date range, "
            "frequency aggregation, and unit transformation. "
            "Prefer over get_macro_indicator when you need a specific date range, "
            "percent-change units, or frequency downsampling. "
            "units: lin=levels, chg=change, ch1=change from year ago, "
            "pch=percent change, pc1=percent change from year ago, log=natural log. "
            "frequency: d=daily, w=weekly, m=monthly, q=quarterly, a=annual."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "series_id": {
                    "type": "string",
                    "description": "FRED series ID, e.g. 'GDP', 'CPIAUCSL', 'INDPRO'.",
                },
                "observation_start": {
                    "type": "string",
                    "description": "Start date in YYYY-MM-DD format (optional).",
                },
                "observation_end": {
                    "type": "string",
                    "description": "End date in YYYY-MM-DD format (optional).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of observations (default 100).",
                    "default": 100,
                },
                "units": {
                    "type": "string",
                    "description": "lin, chg, ch1, pch, pc1, pca, cch, cca, or log (default lin).",
                    "default": "lin",
                    "enum": ["lin", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "log"],
                },
                "frequency": {
                    "type": "string",
                    "description": "Aggregate to: d, w, bw, m, q, sa, a. Omit for native frequency.",
                },
            },
            "required": ["series_id"],
        },
    },
}

FRED_RELEASES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "list_fred_releases",
        "description": (
            "List all FRED economic data releases (named reports that contain series). "
            "Returns release_id, name, and link for each release. "
            "Use release_id with get_fred_release_series to see the series inside a release. "
            "Notable IDs: 10=BLS Employment Situation, 11=PPI, 21=Industrial Production, "
            "53=GDP, 86=CPI, 175=ISM Manufacturing, 184=Durable Goods, 205=Retail Sales."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of releases to return (1–1000, default 50).",
                    "default": 50,
                },
            },
            "required": [],
        },
    },
}

FRED_RELEASE_SERIES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_fred_release_series",
        "description": (
            "Get the economic data series that belong to a specific FRED release, sorted by popularity. "
            "Also returns the release name and link. "
            "Use list_fred_releases to find release_ids. "
            "Notable IDs: 10=BLS Employment Situation, 21=Industrial Production, "
            "53=GDP, 86=CPI, 175=ISM Manufacturing, 184=Durable Goods Orders."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "release_id": {
                    "type": "integer",
                    "description": "FRED release ID (integer).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of series to return (1–100, default 20).",
                    "default": 20,
                },
            },
            "required": ["release_id"],
        },
    },
}

FRED_CATEGORY_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "browse_fred_category",
        "description": (
            "Browse the FRED category hierarchy to discover series by economic topic. "
            "Returns the category name, child categories (with IDs), and optionally "
            "the most popular series within the category. "
            "Start with category_id=0 (root) to see top-level topics, then drill into children. "
            "Key IDs: 0=root, 32991=Money Banking&Finance, 32992=Population/Employment/Labor, "
            "10=Business Cycles, 32262=Production & Business Activity, 32455=Prices, "
            "33936=Trade & International Transactions. "
            "Set include_series=true once you've found the right category."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category_id": {
                    "type": "integer",
                    "description": "FRED category ID. Use 0 for the root.",
                    "default": 0,
                },
                "include_series": {
                    "type": "boolean",
                    "description": "If true, also return the most popular series in this category.",
                    "default": False,
                },
                "series_limit": {
                    "type": "integer",
                    "description": "Number of series to return when include_series=true (default 20).",
                    "default": 20,
                },
            },
            "required": [],
        },
    },
}

FRED_TAGS_SERIES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_fred_series_by_tags",
        "description": (
            "Find FRED economic series matching ALL of a set of topic tags, sorted by popularity. "
            "Semicolon-separate multiple tags: e.g. 'manufacturing;monthly;sa'. "
            "Common topic tags: manufacturing, construction, trade, employment, price, gdp. "
            "Frequency tags: daily, weekly, monthly, quarterly, annual. "
            "Adjustment: sa (seasonally adjusted), nsa (not seasonally adjusted). "
            "Use search_fred_series for free-text search; use this for structured tag filtering."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tag_names": {
                    "type": "string",
                    "description": "Semicolon-separated FRED tag names (all must match).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of series to return (1–100, default 20).",
                    "default": 20,
                },
            },
            "required": ["tag_names"],
        },
    },
}

FRED_SERIES_UPDATES_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "get_fred_series_updates",
        "description": (
            "Get FRED economic series that were recently updated or newly released, "
            "sorted by update timestamp descending. "
            "Useful for tracking which economic data reports were published recently. "
            "filter_value=macro (default) limits to national indicators; "
            "regional includes state/metro series; all returns both."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Number of recently updated series to return (1–100, default 20).",
                    "default": 20,
                },
                "filter_value": {
                    "type": "string",
                    "description": "macro, regional, or all (default macro).",
                    "default": "macro",
                    "enum": ["macro", "regional", "all"],
                },
            },
            "required": [],
        },
    },
}

# All schemas, tagged with which premium tier they require (None = always available).
_FMP_PREMIUM_SCHEMAS = {
    "get_income_statement",
    "get_balance_sheet",
    "get_cash_flow",
    "get_financial_ratios",
    "get_analyst_estimates",
    "get_stock_peers",
    "get_company_rating",
    "get_earnings_surprises",
    "get_press_releases",
    "screen_stocks",
}
_AV_PREMIUM_SCHEMAS = {
    "get_news_sentiment",
    "get_earnings_transcript",
    "get_insider_transactions",
}

_ALL_TOOL_SCHEMAS: list[dict] = [
    NEWS_SEARCH_SCHEMA,
    NEWS_TOP_HEADLINES_SCHEMA,
    NEWS_SOURCES_SCHEMA,
    MINING_METALS_PRICES_SCHEMA,
    ENERGY_COST_PRICES_SCHEMA,
    BROAD_COMMODITY_CYCLE_SCHEMA,
    AGRICULTURAL_COMMODITY_PRICES_SCHEMA,
    FX_RATES_SCHEMA,
    COMPANY_FINANCIALS_SCHEMA,
    SEC_FILINGS_SCHEMA,
    EQUITY_PRICE_SCHEMA,
    EQUITY_HISTORY_SCHEMA,
    EQUITY_FINANCIALS_SCHEMA,
    EARNINGS_CALENDAR_SCHEMA,
    NEWS_SENTIMENT_SCHEMA,
    EARNINGS_TRANSCRIPT_SCHEMA,
    INSIDER_TRANSACTIONS_SCHEMA,
    WEB_SEARCH_SCHEMA,
    WEB_EXTRACT_SCHEMA,
    WEB_CRAWL_SCHEMA,
    WEB_MAP_SCHEMA,
    WEB_RESEARCH_SCHEMA,
    MASTERDATA_LOOKUP_SCHEMA,
    MACRO_INDICATORS_SCHEMA,
    FRED_SEARCH_SERIES_SCHEMA,
    FRED_OBSERVATIONS_SCHEMA,
    FRED_RELEASES_SCHEMA,
    FRED_RELEASE_SERIES_SCHEMA,
    FRED_CATEGORY_SCHEMA,
    FRED_TAGS_SERIES_SCHEMA,
    FRED_SERIES_UPDATES_SCHEMA,
    INCOME_STATEMENT_SCHEMA,
    BALANCE_SHEET_SCHEMA,
    CASH_FLOW_SCHEMA,
    FINANCIAL_RATIOS_SCHEMA,
    ANALYST_ESTIMATES_SCHEMA,
    STOCK_PEERS_SCHEMA,
    COMPANY_RATING_SCHEMA,
    EARNINGS_SURPRISES_SCHEMA,
    PRESS_RELEASES_SCHEMA,
    STOCK_SCREENER_SCHEMA,
    TECHNICAL_REPORT_SCHEMA,
]


def _schema_tier_filter(schemas: list[dict]) -> list[dict]:
    """Drop schemas for tools that require a premium subscription when tier is 'free'."""
    keep = []
    for s in schemas:
        name = s.get("function", {}).get("name", "")
        if name in _FMP_PREMIUM_SCHEMAS and _settings.fmp_tier == "free":
            continue
        if name in _AV_PREMIUM_SCHEMAS and _settings.alpha_vantage_tier == "free":
            continue
        keep.append(s)
    return keep


TOOL_SCHEMAS: list[dict] = _schema_tier_filter(_ALL_TOOL_SCHEMAS)

INDUSTRY_KNOWLEDGE_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_industry_knowledge",
        "description": (
            "Search the industry knowledge base of mining, construction, and heavy equipment "
            "books and articles. Use to understand what data is needed to answer a question "
            "correctly, or to interpret what collected signals mean for Komatsu."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing what knowledge you need.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of knowledge chunks to return (default 5).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}

EPISODIC_MEMORY_SCHEMA: dict = {
    "type": "function",
    "function": {
        "name": "search_episodic_memory",
        "description": (
            "Search past successful research reports and their execution plans. "
            "Use during planning to find examples of plans that worked for similar questions. "
            "Use during synthesis to see how past reports handled similar signals and conclusions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query describing the type of past plan or report you need.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of past plan chunks to return (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}

# Stage-specific schema lists consumed by graph nodes via tool_router
UNDERSTAND_SCHEMAS: list[dict] = [
    INDUSTRY_KNOWLEDGE_SCHEMA,
    EPISODIC_MEMORY_SCHEMA,
    WEB_SEARCH_SCHEMA,
    NEWS_SEARCH_SCHEMA,
    NEWS_TOP_HEADLINES_SCHEMA,
    WEB_EXTRACT_SCHEMA,
    MASTERDATA_LOOKUP_SCHEMA,
]
# Research loop uses a focused subset (no RAG tools — those are for planning context)
RESEARCH_SCHEMAS: list[dict] = [
    WEB_SEARCH_SCHEMA,
    NEWS_SEARCH_SCHEMA,
    NEWS_TOP_HEADLINES_SCHEMA,
    WEB_EXTRACT_SCHEMA,
    MASTERDATA_LOOKUP_SCHEMA,
]
SYNTHESIZE_SCHEMAS: list[dict] = [INDUSTRY_KNOWLEDGE_SCHEMA, EPISODIC_MEMORY_SCHEMA]
COLLECT_SCHEMAS: list[dict] = TOOL_SCHEMAS
