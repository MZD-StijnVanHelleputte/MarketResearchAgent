from config import settings
from clients.base_http_client import BaseHttpClient


class AlphaVantageClient(BaseHttpClient):
    """Raw Alpha Vantage client — returns dicts, no business logic."""

    def __init__(self) -> None:
        super().__init__(
            base_url=settings.alpha_vantage_base_url,
            api_key="",  # AV uses apikey query param, not Authorization header
            timeout_s=settings.alpha_vantage_timeout_s,
            max_retries=settings.alpha_vantage_max_retries,
            rate_limit_per_min=settings.alpha_vantage_rate_limit_per_min,
        )
        self._api_key = settings.alpha_vantage_api_key

    def _auth_headers(self) -> dict:
        return {}  # key travels as the apikey query param

    async def get_global_quote(self, symbol: str) -> dict:
        """GLOBAL_QUOTE — works for equities/ETFs only (not raw commodities)."""
        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self._api_key,
        }
        return await self.get("/query", params=params)

    async def get_gold_silver_spot(self, symbol: str) -> dict:
        """GOLD_SILVER_SPOT — live spot prices for GOLD/XAU and SILVER/XAG."""
        params = {
            "function": "GOLD_SILVER_SPOT",
            "symbol": symbol,
            "apikey": self._api_key,
        }
        return await self.get("/query", params=params)

    async def get_gold_silver_history(self, symbol: str, interval: str = "monthly") -> dict:
        """GOLD_SILVER_HISTORY — historical GOLD/XAU and SILVER/XAG prices."""
        params = {
            "function": "GOLD_SILVER_HISTORY",
            "symbol": symbol,
            "interval": interval,
            "apikey": self._api_key,
        }
        return await self.get("/query", params=params)

    async def get_commodity_series(self, function: str, interval: str = "monthly") -> dict:
        """Dedicated commodity endpoint.

        Current tool surface uses WTI, BRENT, NATURAL_GAS, COPPER, ALUMINUM,
        ALL_COMMODITIES, WHEAT, and CORN. Other Alpha Vantage commodity functions
        may still pass through here when explicitly supported by a service.

        Returns {"name", "interval", "unit", "data": [{"date", "value"}, ...]} on
        success, or an {"Information"|"Note"|"Error Message": ...} body (HTTP 200)
        when the free-tier daily cap (25 req/day) is hit.
        """
        params = {
            "function": function,
            "interval": interval,
            "apikey": self._api_key,
        }
        return await self.get("/query", params=params)

    async def get_fx_series(
        self, from_symbol: str, to_symbol: str, function: str
    ) -> dict:
        """FX_DAILY | FX_WEEKLY | FX_MONTHLY — historical exchange-rate series."""
        params = {
            "function": function,
            "from_symbol": from_symbol,
            "to_symbol": to_symbol,
            "outputsize": "compact",
            "apikey": self._api_key,
        }
        return await self.get("/query", params=params)

    async def get_news_sentiment(
        self,
        tickers: str = "",
        topics: str = "",
        sort: str = "LATEST",
        limit: int = 25,
    ) -> dict:
        """NEWS_SENTIMENT — sentiment-scored news articles (Alpha Intelligence; premium)."""
        params: dict = {
            "function": "NEWS_SENTIMENT",
            "sort": sort,
            "limit": str(limit),
            "apikey": self._api_key,
        }
        if tickers:
            params["tickers"] = tickers
        if topics:
            params["topics"] = topics
        return await self.get("/query", params=params)

    async def get_earnings_transcript(self, symbol: str, quarter: str) -> dict:
        """EARNINGS_CALL_TRANSCRIPT — parsed earnings call text (premium)."""
        params = {
            "function": "EARNINGS_CALL_TRANSCRIPT",
            "symbol": symbol,
            "quarter": quarter,
            "apikey": self._api_key,
        }
        return await self.get("/query", params=params)

    async def get_insider_transactions(self, symbol: str) -> dict:
        """INSIDER_TRANSACTIONS — recent insider buying/selling activity (premium)."""
        params = {
            "function": "INSIDER_TRANSACTIONS",
            "symbol": symbol,
            "apikey": self._api_key,
        }
        return await self.get("/query", params=params)

    async def get_earnings_calendar(
        self, symbol: str = "", horizon: str = "3month"
    ) -> dict:
        """EARNINGS_CALENDAR — upcoming earnings announcement dates (free tier)."""
        params: dict = {
            "function": "EARNINGS_CALENDAR",
            "horizon": horizon,
            "apikey": self._api_key,
        }
        if symbol:
            params["symbol"] = symbol
        return await self.get("/query", params=params)
