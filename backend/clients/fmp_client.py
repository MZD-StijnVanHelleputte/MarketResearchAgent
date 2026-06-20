from config import settings
from clients.base_http_client import BaseHttpClient


class FmpClient(BaseHttpClient):
    """Raw Financial Modeling Prep client (/stable/ API) — returns dicts, no business logic.

    The free tier only serves /stable/profile; the statement/metrics endpoints return
    HTTP 402 (premium). Callers should treat those as best-effort.
    """

    def __init__(self) -> None:
        super().__init__(
            base_url=settings.fmp_base_url,
            api_key="",  # FMP uses apikey query param, not Authorization header
            timeout_s=settings.fmp_timeout_s,
            max_retries=settings.fmp_max_retries,
            rate_limit_per_min=settings.fmp_rate_limit_per_min,
        )
        self._api_key = settings.fmp_api_key

    def _auth_headers(self) -> dict:
        return {}  # key travels as the apikey query param

    async def get_company_profile(self, ticker: str) -> dict:
        """GET /stable/profile?symbol= — company overview, price and market cap (free tier)."""
        return await self.get("/stable/profile", params={"symbol": ticker, "apikey": self._api_key})

    async def get_key_metrics(self, ticker: str) -> dict:
        """GET /stable/key-metrics-ttm?symbol= — TTM key metrics (premium-gated; 402 on free)."""
        return await self.get(
            "/stable/key-metrics-ttm", params={"symbol": ticker, "apikey": self._api_key}
        )

    async def get_income_statement(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[dict]:
        """GET /stable/income-statement — revenue, gross profit, operating income, net income, EPS."""
        return await self.get(
            "/stable/income-statement",
            params={"symbol": ticker, "period": period, "limit": limit, "apikey": self._api_key},
        )

    async def get_balance_sheet(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[dict]:
        """GET /stable/balance-sheet-statement — assets, liabilities, equity, cash, net debt."""
        return await self.get(
            "/stable/balance-sheet-statement",
            params={"symbol": ticker, "period": period, "limit": limit, "apikey": self._api_key},
        )

    async def get_cash_flow(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[dict]:
        """GET /stable/cash-flow-statement — operating CF, capex, free CF, financing activities."""
        return await self.get(
            "/stable/cash-flow-statement",
            params={"symbol": ticker, "period": period, "limit": limit, "apikey": self._api_key},
        )

    async def get_ratios(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[dict]:
        """GET /stable/ratios — PE, EV/EBITDA, debt/equity, ROE, ROIC, current ratio."""
        return await self.get(
            "/stable/ratios",
            params={"symbol": ticker, "period": period, "limit": limit, "apikey": self._api_key},
        )

    async def get_analyst_estimates(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[dict]:
        """GET /stable/analyst-estimates — consensus revenue and EPS estimates."""
        return await self.get(
            "/stable/analyst-estimates",
            params={"symbol": ticker, "period": period, "limit": limit, "apikey": self._api_key},
        )

    async def get_stock_peers(self, ticker: str) -> list[dict]:
        """GET /stable/stock-peers — peer companies for competitive benchmarking."""
        return await self.get(
            "/stable/stock-peers", params={"symbol": ticker, "apikey": self._api_key}
        )

    async def get_company_rating(self, ticker: str) -> list[dict]:
        """GET /stable/rating — DCF-based overall buy/sell rating and sub-scores."""
        return await self.get(
            "/stable/rating", params={"symbol": ticker, "apikey": self._api_key}
        )

    async def get_earnings_surprises(self, ticker: str, limit: int = 8) -> list[dict]:
        """GET /stable/earnings-surprises — actual vs estimated EPS beat/miss history."""
        return await self.get(
            "/stable/earnings-surprises",
            params={"symbol": ticker, "limit": limit, "apikey": self._api_key},
        )

    async def get_press_releases(self, ticker: str, limit: int = 10) -> list[dict]:
        """GET /stable/press-releases — official corporate press releases."""
        return await self.get(
            "/stable/press-releases",
            params={"symbol": ticker, "limit": limit, "apikey": self._api_key},
        )

    async def screen_stocks(
        self,
        sector: str | None = None,
        industry: str | None = None,
        country: str | None = None,
        market_cap_min: float | None = None,
        market_cap_max: float | None = None,
        exchange: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """GET /stable/screener — filter companies by sector, country, market cap, exchange."""
        params: dict = {"limit": limit, "apikey": self._api_key}
        if sector:
            params["sector"] = sector
        if industry:
            params["industry"] = industry
        if country:
            params["country"] = country
        if market_cap_min is not None:
            params["marketCapMoreThan"] = market_cap_min
        if market_cap_max is not None:
            params["marketCapLowerThan"] = market_cap_max
        if exchange:
            params["exchange"] = exchange
        return await self.get("/stable/company-screener", params=params)
