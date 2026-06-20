import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class CashFlowInput(BaseModel):
    ticker: str
    period: str = Field(default="annual", description="'annual' or 'quarter'")
    limit: int = Field(default=8, ge=1, le=20)


class CashFlowTool(BaseTool):
    name = "get_cash_flow"
    requires_premium = "fmp"
    description = (
        "Get historical cash flow statements for a publicly traded company. "
        "Returns operating cash flow, capital expenditure (capex), and free cash flow "
        "for up to 20 annual or quarterly periods. Use for equipment investment cycle signals."
    )
    input_schema = CashFlowInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = CashFlowInput(**kwargs)
        results = await self._service.get_cash_flow(
            inp.ticker, period=inp.period, limit=inp.limit
        )
        return {"statements": [dataclasses.asdict(r) for r in results]}
