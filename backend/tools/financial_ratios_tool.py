import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class FinancialRatiosInput(BaseModel):
    ticker: str
    period: str = Field(default="annual", description="'annual' or 'quarter'")
    limit: int = Field(default=8, ge=1, le=20)


class FinancialRatiosTool(BaseTool):
    name = "get_financial_ratios"
    requires_premium = "fmp"
    description = (
        "Get historical financial ratios for a publicly traded company. "
        "Returns P/E ratio, EV/EBITDA, debt-to-equity, ROE, ROIC, and current ratio "
        "for up to 20 annual or quarterly periods. Use for valuation benchmarking."
    )
    input_schema = FinancialRatiosInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = FinancialRatiosInput(**kwargs)
        results = await self._service.get_financial_ratios(
            inp.ticker, period=inp.period, limit=inp.limit
        )
        return {"ratios": [dataclasses.asdict(r) for r in results]}
