import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class EarningsSurprisesInput(BaseModel):
    ticker: str
    limit: int = Field(default=8, ge=1, le=40)


class EarningsSurprisesTool(BaseTool):
    name = "get_earnings_surprises"
    requires_premium = "fmp"
    description = (
        "Get the earnings surprise history for a publicly traded company. "
        "Returns actual EPS vs analyst consensus estimate and the surprise percentage "
        "for the most recent quarterly reports. Use to assess management execution quality."
    )
    input_schema = EarningsSurprisesInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = EarningsSurprisesInput(**kwargs)
        results = await self._service.get_earnings_surprises(inp.ticker, limit=inp.limit)
        return {"surprises": [dataclasses.asdict(r) for r in results]}
