import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class BalanceSheetInput(BaseModel):
    ticker: str
    period: str = Field(default="annual", description="'annual' or 'quarter'")
    limit: int = Field(default=8, ge=1, le=20)


class BalanceSheetTool(BaseTool):
    name = "get_balance_sheet"
    requires_premium = "fmp"
    description = (
        "Get historical balance sheets for a publicly traded company. "
        "Returns cash, total assets, total debt, total equity, and net debt "
        "for up to 20 annual or quarterly periods."
    )
    input_schema = BalanceSheetInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = BalanceSheetInput(**kwargs)
        results = await self._service.get_balance_sheet(
            inp.ticker, period=inp.period, limit=inp.limit
        )
        return {"statements": [dataclasses.asdict(r) for r in results]}
