import logging
from dataclasses import dataclass
from datetime import date
from config import settings
from clients.fmp_client import FmpClient
from clients.edgar_client import EdgarClient
from clients.yfinance_client import YFinanceClient
from clients.base_http_client import ClientError

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    pass


@dataclass
class CompanyFinancials:
    ticker: str
    name: str
    price_usd: float | None
    market_cap_usd: float | None
    revenue_usd: float | None
    net_income_usd: float | None
    capex_usd: float | None
    pe_ratio: float | None
    currency: str
    industry: str
    date: str


@dataclass
class Filing:
    entity_name: str
    form_type: str
    file_date: str
    period: str
    snippet: str


def _strip_cik(display_name: str) -> str:
    """'APPLE INC (AAPL) (CIK 0000320193)' -> 'APPLE INC (AAPL)'."""
    return display_name.split("(CIK")[0].strip()


class CompetitionService:
    """Business logic for competitor financials and SEC filings."""

    def __init__(
        self,
        fmp_client: FmpClient | None = None,
        edgar_client: EdgarClient | None = None,
        yf_client: YFinanceClient | None = None,
    ) -> None:
        self._fmp = fmp_client or FmpClient()
        self._edgar = edgar_client or EdgarClient()
        self._yf = yf_client or YFinanceClient()

    async def get_financials(self, ticker: str) -> CompanyFinancials:
        # On the FMP free tier only /stable/profile works and the metrics fields stay empty,
        # so yfinance (free, no rate limit, richer data) is the primary source. On premium,
        # FMP leads and yfinance is the fallback. Each path falls back to the other on failure.
        if settings.fmp_tier == "premium":
            try:
                return await self._financials_from_fmp(ticker)
            except (ClientError, ServiceError) as exc:
                logger.warning("FMP financials failed for %s, falling back to yfinance: %s", ticker, exc)
                return await self._financials_from_yfinance(ticker)

        try:
            return await self._financials_from_yfinance(ticker)
        except ServiceError as exc:
            logger.warning("yfinance financials failed for %s, falling back to FMP: %s", ticker, exc)
            return await self._financials_from_fmp(ticker)

    async def _financials_from_yfinance(self, ticker: str) -> CompanyFinancials:
        try:
            o = await self._yf.get_company_overview(ticker)
        except Exception as exc:
            raise ServiceError(f"yfinance overview failed for {ticker}: {exc}") from exc

        if not o.get("name") and o.get("price") is None and o.get("market_cap") is None:
            raise ServiceError(f"yfinance returned no data for {ticker}")

        return CompanyFinancials(
            ticker=ticker.upper(),
            name=o.get("name") or ticker,
            price_usd=o.get("price"),
            market_cap_usd=o.get("market_cap"),
            revenue_usd=o.get("revenue"),
            net_income_usd=o.get("net_income"),
            capex_usd=o.get("capex"),
            pe_ratio=o.get("pe_ratio"),
            currency=o.get("currency") or "USD",
            industry=o.get("industry") or "",
            date=o.get("date") or date.today().isoformat(),
        )

    async def _financials_from_fmp(self, ticker: str) -> CompanyFinancials:
        # /stable/profile is the only endpoint on the FMP free tier; statement/metrics
        # endpoints return HTTP 402, so those are attempted best-effort.
        try:
            profile_list = await self._fmp.get_company_profile(ticker)
        except ClientError as exc:
            raise ServiceError(f"FMP profile request failed for {ticker}: {exc}") from exc

        profile = profile_list[0] if isinstance(profile_list, list) and profile_list else {}
        if not profile:
            raise ServiceError(f"FMP returned no profile for {ticker}")

        metrics: dict = {}
        try:
            metrics_list = await self._fmp.get_key_metrics(ticker)
            metrics = metrics_list[0] if isinstance(metrics_list, list) and metrics_list else {}
        except ClientError as exc:
            logger.info("FMP key-metrics unavailable for %s (premium-gated): %s", ticker, exc)

        return CompanyFinancials(
            ticker=ticker.upper(),
            name=profile.get("companyName") or ticker,
            price_usd=profile.get("price"),
            market_cap_usd=profile.get("marketCap"),
            revenue_usd=metrics.get("revenueTTM"),
            net_income_usd=metrics.get("netIncomeTTM"),
            capex_usd=metrics.get("capexTTM"),
            pe_ratio=metrics.get("peRatioTTM"),
            currency=profile.get("currency") or "USD",
            industry=profile.get("industry") or "",
            date=date.today().isoformat(),
        )

    async def get_filings(
        self, query: str, forms: str = "10-K,10-Q,8-K", limit: int = 25
    ) -> list[Filing]:
        try:
            raw = await self._edgar.search_filings(query=query, forms=forms)
        except ClientError as exc:
            raise ServiceError(f"EDGAR request failed: {exc}") from exc

        hits = raw.get("hits", {}).get("hits", [])[:limit]
        filings: list[Filing] = []
        for hit in hits:
            src = hit.get("_source", {})
            names = src.get("display_names") or []
            root_forms = src.get("root_forms") or []
            filings.append(Filing(
                entity_name=_strip_cik(names[0]) if names else "",
                form_type=src.get("form") or (root_forms[0] if root_forms else ""),
                file_date=src.get("file_date") or "",
                period=src.get("period_ending") or "",
                snippet=(src.get("file_type") or "") + (f" — {hit.get('_id', '')}" if hit.get("_id") else ""),
            ))
        return filings
