import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class StockScreenerInput(BaseModel):
    sector: str | None = None
    industry: str | None = None
    country: str | None = None
    market_cap_min: float | None = Field(default=None, description="Minimum market cap in USD")
    market_cap_max: float | None = Field(default=None, description="Maximum market cap in USD")
    exchange: str | None = Field(
        default=None, description="Exchange code, e.g. NYSE, NASDAQ, TSX, EURONEXT"
    )
    limit: int = Field(default=20, ge=1, le=100)


class StockScreenerTool(BaseTool):
    name = "screen_stocks"
    requires_premium = "fmp"
    description = (
        "Screen for publicly traded companies matching specific criteria. "
        "Filter by sector, industry, country, market cap range, or exchange. "
        "Returns ticker, company name, price, market cap, sector, industry, country, and exchange. "
        "Use to discover unknown competitors, customers, or mining project developers."
    )
    input_schema = StockScreenerInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = StockScreenerInput(**kwargs)
        results = await self._service.screen_stocks(
            sector=inp.sector,
            industry=inp.industry,
            country=inp.country,
            market_cap_min=inp.market_cap_min,
            market_cap_max=inp.market_cap_max,
            exchange=inp.exchange,
            limit=inp.limit,
        )
        return {"results": [dataclasses.asdict(r) for r in results]}
