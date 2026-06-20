import dataclasses
from typing import Literal
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.equity_service import EquityService


class EquityFinancialsInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'CAT', 'VOLV-B.ST', '6305.T'.")
    period: Literal["annual", "quarterly"] = Field(
        default="annual",
        description=(
            "Statement granularity. 'annual' returns ~4 fiscal years of income-statement "
            "line items; 'quarterly' returns ~4 most recent quarters. Use this for "
            "year-over-year or quarter-over-quarter trend analysis when FMP fundamentals "
            "are unavailable."
        ),
    )


class EquityFinancialsTool(BaseTool):
    name = "get_equity_financials"
    description = (
        "Get multi-year (annual) or multi-quarter income-statement line items "
        "(revenue, net income, operating income, etc.) for a stock ticker via Yahoo "
        "Finance. Free, no premium tier required — use this for long-term financial "
        "trend analysis of competitors when FMP fundamentals tools are unavailable. "
        "Returns ticker, period, and a list of rows (one per fiscal period) with date "
        "and line-item values."
    )
    input_schema = EquityFinancialsInput

    def __init__(self, service: EquityService | None = None) -> None:
        self._service = service or EquityService()

    async def run(self, **kwargs) -> dict:
        inp = EquityFinancialsInput(**kwargs)
        result = await self._service.get_financials(inp.ticker, inp.period)
        return dataclasses.asdict(result)
