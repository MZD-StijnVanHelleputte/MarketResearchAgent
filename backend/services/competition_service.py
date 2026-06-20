import logging
from dataclasses import dataclass
from datetime import date
from clients.fmp_client import FmpClient
from clients.edgar_client import EdgarClient
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
    ) -> None:
        self._fmp = fmp_client or FmpClient()
        self._edgar = edgar_client or EdgarClient()

    async def get_financials(self, ticker: str) -> CompanyFinancials:
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
        self, query: str, forms: str = "10-K,10-Q,8-K", limit: int = 10
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
