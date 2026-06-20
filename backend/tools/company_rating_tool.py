import dataclasses
from pydantic import BaseModel
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class CompanyRatingInput(BaseModel):
    ticker: str


class CompanyRatingTool(BaseTool):
    name = "get_company_rating"
    requires_premium = "fmp"
    description = (
        "Get the overall DCF-based investment rating for a publicly traded company. "
        "Returns a rating label (Strong Buy / Buy / Neutral / Sell / Strong Sell), "
        "an overall score, and sub-scores for DCF, ROE, and debt. "
        "Use for a quick valuation health check on competitors or customers."
    )
    input_schema = CompanyRatingInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = CompanyRatingInput(**kwargs)
        results = await self._service.get_company_rating(inp.ticker)
        return {"ratings": [dataclasses.asdict(r) for r in results]}
