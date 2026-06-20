import logging
from dataclasses import dataclass
from clients.fmp_client import FmpClient
from clients.base_http_client import ClientError

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    pass


@dataclass
class IncomeStatement:
    ticker: str
    period: str
    date: str
    revenue: float | None
    gross_profit: float | None
    operating_income: float | None
    net_income: float | None
    ebitda: float | None
    eps: float | None


@dataclass
class BalanceSheet:
    ticker: str
    period: str
    date: str
    cash: float | None
    total_assets: float | None
    total_debt: float | None
    total_equity: float | None
    net_debt: float | None


@dataclass
class CashFlow:
    ticker: str
    period: str
    date: str
    operating_cf: float | None
    capex: float | None
    free_cf: float | None
    net_cash: float | None


@dataclass
class FinancialRatios:
    ticker: str
    period: str
    date: str
    pe_ratio: float | None
    ev_ebitda: float | None
    debt_equity: float | None
    roe: float | None
    roic: float | None
    current_ratio: float | None


@dataclass
class AnalystEstimate:
    ticker: str
    period: str
    date: str
    est_revenue_avg: float | None
    est_revenue_high: float | None
    est_revenue_low: float | None
    est_eps_avg: float | None
    num_analysts: int | None


@dataclass
class StockPeers:
    ticker: str
    peers: list[str]


@dataclass
class CompanyRating:
    ticker: str
    date: str
    rating: str | None
    score: int | None
    dcf_score: int | None
    roe_score: int | None
    debt_score: int | None


@dataclass
class EarningsSurprise:
    ticker: str
    date: str
    actual_eps: float | None
    estimated_eps: float | None
    surprise_pct: float | None


@dataclass
class PressRelease:
    ticker: str
    title: str
    date: str
    content: str


@dataclass
class ScreenResult:
    ticker: str
    name: str | None
    market_cap: float | None
    price: float | None
    sector: str | None
    industry: str | None
    country: str | None
    exchange: str | None


class FundamentalsService:
    """Business logic for FMP fundamental data endpoints (statements, ratios, estimates, etc.)."""

    def __init__(self, fmp_client: FmpClient | None = None) -> None:
        self._fmp = fmp_client or FmpClient()

    async def get_income_statement(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[IncomeStatement]:
        try:
            raw = await self._fmp.get_income_statement(ticker, period=period, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FMP income-statement failed for {ticker}: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        return [
            IncomeStatement(
                ticker=ticker.upper(),
                period=r.get("period") or period,
                date=r.get("date") or "",
                revenue=r.get("revenue"),
                gross_profit=r.get("grossProfit"),
                operating_income=r.get("operatingIncome"),
                net_income=r.get("netIncome"),
                ebitda=r.get("ebitda"),
                eps=r.get("eps"),
            )
            for r in items
        ]

    async def get_balance_sheet(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[BalanceSheet]:
        try:
            raw = await self._fmp.get_balance_sheet(ticker, period=period, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FMP balance-sheet failed for {ticker}: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        return [
            BalanceSheet(
                ticker=ticker.upper(),
                period=r.get("period") or period,
                date=r.get("date") or "",
                cash=r.get("cashAndCashEquivalents"),
                total_assets=r.get("totalAssets"),
                total_debt=r.get("totalDebt"),
                total_equity=r.get("totalStockholdersEquity"),
                net_debt=r.get("netDebt"),
            )
            for r in items
        ]

    async def get_cash_flow(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[CashFlow]:
        try:
            raw = await self._fmp.get_cash_flow(ticker, period=period, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FMP cash-flow failed for {ticker}: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        return [
            CashFlow(
                ticker=ticker.upper(),
                period=r.get("period") or period,
                date=r.get("date") or "",
                operating_cf=r.get("operatingCashFlow"),
                capex=r.get("capitalExpenditure"),
                free_cf=r.get("freeCashFlow"),
                net_cash=r.get("netCashProvidedByOperatingActivities"),
            )
            for r in items
        ]

    async def get_financial_ratios(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[FinancialRatios]:
        try:
            raw = await self._fmp.get_ratios(ticker, period=period, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FMP ratios failed for {ticker}: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        return [
            FinancialRatios(
                ticker=ticker.upper(),
                period=r.get("period") or period,
                date=r.get("date") or "",
                pe_ratio=r.get("priceEarningsRatio"),
                ev_ebitda=r.get("enterpriseValueMultiple"),
                debt_equity=r.get("debtEquityRatio"),
                roe=r.get("returnOnEquity"),
                roic=r.get("returnOnCapitalEmployed"),
                current_ratio=r.get("currentRatio"),
            )
            for r in items
        ]

    async def get_analyst_estimates(
        self, ticker: str, period: str = "annual", limit: int = 4
    ) -> list[AnalystEstimate]:
        try:
            raw = await self._fmp.get_analyst_estimates(ticker, period=period, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FMP analyst-estimates failed for {ticker}: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        return [
            AnalystEstimate(
                ticker=ticker.upper(),
                period=r.get("period") or period,
                date=r.get("date") or "",
                est_revenue_avg=r.get("estimatedRevenueAvg"),
                est_revenue_high=r.get("estimatedRevenueHigh"),
                est_revenue_low=r.get("estimatedRevenueLow"),
                est_eps_avg=r.get("estimatedEpsAvg"),
                num_analysts=r.get("numberAnalystEstimatedRevenue"),
            )
            for r in items
        ]

    async def get_stock_peers(self, ticker: str) -> StockPeers:
        try:
            raw = await self._fmp.get_stock_peers(ticker)
        except ClientError as exc:
            raise ServiceError(f"FMP stock-peers failed for {ticker}: {exc}") from exc
        # /stable/stock-peers returns a flat list of peer company objects: [{symbol, companyName, ...}]
        items = raw if isinstance(raw, list) else []
        peers = [r["symbol"] for r in items if r.get("symbol")]
        return StockPeers(ticker=ticker.upper(), peers=peers)

    async def get_company_rating(self, ticker: str) -> list[CompanyRating]:
        try:
            raw = await self._fmp.get_company_rating(ticker)
        except ClientError as exc:
            raise ServiceError(f"FMP rating failed for {ticker}: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        return [
            CompanyRating(
                ticker=ticker.upper(),
                date=r.get("date") or "",
                rating=r.get("rating"),
                score=r.get("ratingScore"),
                dcf_score=r.get("ratingDetailsDCFScore"),
                roe_score=r.get("ratingDetailsROEScore"),
                debt_score=r.get("ratingDetailsDebtToEquityScore"),
            )
            for r in items
        ]

    async def get_earnings_surprises(
        self, ticker: str, limit: int = 8
    ) -> list[EarningsSurprise]:
        try:
            raw = await self._fmp.get_earnings_surprises(ticker, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FMP earnings-surprises failed for {ticker}: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        results = []
        for r in items:
            actual = r.get("actualEarningResult")
            estimated = r.get("estimatedEarning")
            pct: float | None = None
            if actual is not None and estimated and estimated != 0:
                pct = round((actual - estimated) / abs(estimated) * 100, 2)
            results.append(
                EarningsSurprise(
                    ticker=ticker.upper(),
                    date=r.get("date") or "",
                    actual_eps=actual,
                    estimated_eps=estimated,
                    surprise_pct=pct,
                )
            )
        return results

    async def get_press_releases(
        self, ticker: str, limit: int = 10
    ) -> list[PressRelease]:
        try:
            raw = await self._fmp.get_press_releases(ticker, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FMP press-releases failed for {ticker}: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        return [
            PressRelease(
                ticker=ticker.upper(),
                title=r.get("title") or "",
                date=r.get("date") or "",
                content=(r.get("text") or "")[:2000],  # cap at 2000 chars to keep response lean
            )
            for r in items
        ]

    async def screen_stocks(
        self,
        sector: str | None = None,
        industry: str | None = None,
        country: str | None = None,
        market_cap_min: float | None = None,
        market_cap_max: float | None = None,
        exchange: str | None = None,
        limit: int = 20,
    ) -> list[ScreenResult]:
        try:
            raw = await self._fmp.screen_stocks(
                sector=sector,
                industry=industry,
                country=country,
                market_cap_min=market_cap_min,
                market_cap_max=market_cap_max,
                exchange=exchange,
                limit=limit,
            )
        except ClientError as exc:
            raise ServiceError(f"FMP screener failed: {exc}") from exc
        items = raw if isinstance(raw, list) else []
        return [
            ScreenResult(
                ticker=r.get("symbol") or "",
                name=r.get("companyName"),
                market_cap=r.get("marketCap"),
                price=r.get("price"),
                sector=r.get("sector"),
                industry=r.get("industry"),
                country=r.get("country"),
                exchange=r.get("exchangeShortName"),
            )
            for r in items
        ]
