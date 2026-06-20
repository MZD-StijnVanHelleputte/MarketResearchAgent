import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class AnalystEstimatesInput(BaseModel):
    ticker: str
    period: str = Field(default="annual", description="'annual' or 'quarter'")
    limit: int = Field(default=8, ge=1, le=20)


class AnalystEstimatesTool(BaseTool):
    name = "get_analyst_estimates"
    requires_premium = "fmp"
    description = (
        "Get Wall Street analyst consensus estimates for a publicly traded company. "
        "Returns forward revenue and EPS estimates (high, low, average) and analyst count. "
        "Use for forward-looking competitor or customer outlook."
    )
    input_schema = AnalystEstimatesInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = AnalystEstimatesInput(**kwargs)
        results = await self._service.get_analyst_estimates(
            inp.ticker, period=inp.period, limit=inp.limit
        )
        return {"estimates": [dataclasses.asdict(r) for r in results]}
