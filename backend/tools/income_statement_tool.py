import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class IncomeStatementInput(BaseModel):
    ticker: str
    period: str = Field(default="annual", description="'annual' or 'quarter'")
    limit: int = Field(default=8, ge=1, le=20)


class IncomeStatementTool(BaseTool):
    name = "get_income_statement"
    requires_premium = "fmp"
    description = (
        "Get historical income statements (P&L) for a publicly traded company. "
        "Returns revenue, gross profit, operating income, net income, EBITDA, and EPS "
        "for up to 20 annual or quarterly periods."
    )
    input_schema = IncomeStatementInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = IncomeStatementInput(**kwargs)
        results = await self._service.get_income_statement(
            inp.ticker, period=inp.period, limit=inp.limit
        )
        return {"statements": [dataclasses.asdict(r) for r in results]}
