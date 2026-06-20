from dataclasses import dataclass, field
from datetime import date
from clients.yfinance_client import YFinanceClient


class ServiceError(Exception):
    pass


@dataclass
class EquityPrice:
    ticker: str
    price: float | None
    currency: str
    market_cap_usd: float | None
    date: str


@dataclass
class EquityHistory:
    ticker: str
    period: str
    rows: list[dict] = field(default_factory=list)


@dataclass
class EquityFinancials:
    ticker: str
    period: str
    rows: list[dict] = field(default_factory=list)


class EquityService:
    """Business logic for equity prices and price history via yfinance."""

    def __init__(self, client: YFinanceClient | None = None) -> None:
        self._client = client or YFinanceClient()

    async def get_price(self, ticker: str) -> EquityPrice:
        try:
            raw = await self._client.get_price(ticker)
        except Exception as exc:
            raise ServiceError(f"yfinance price fetch failed for {ticker}: {exc}") from exc

        return EquityPrice(
            ticker=ticker.upper(),
            price=raw.get("price"),
            currency=raw.get("currency", "USD"),
            market_cap_usd=raw.get("market_cap"),
            date=raw.get("date", date.today().isoformat()),
        )

    async def get_history(self, ticker: str, period: str = "3mo") -> EquityHistory:
        try:
            rows = await self._client.get_history(ticker, period)
        except Exception as exc:
            raise ServiceError(f"yfinance history fetch failed for {ticker}: {exc}") from exc

        return EquityHistory(ticker=ticker.upper(), period=period, rows=rows)

    async def get_financials(self, ticker: str, period: str = "annual") -> EquityFinancials:
        try:
            rows = await self._client.get_financials(ticker, period)
        except Exception as exc:
            raise ServiceError(f"yfinance financials fetch failed for {ticker}: {exc}") from exc

        return EquityFinancials(ticker=ticker.upper(), period=period, rows=rows)
