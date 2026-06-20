import dataclasses
from pydantic import BaseModel
from tools.base import BaseTool
from services.competition_service import CompetitionService


class CompanyFinancialsInput(BaseModel):
    ticker: str  # e.g. "CAT", "6305.T"


class CompanyFinancialsTool(BaseTool):
    name = "get_company_financials"
    description = (
        "Get the latest financial summary for a publicly traded company by ticker symbol. "
        "Returns revenue, net income, capex, market cap, and P/E ratio."
    )
    input_schema = CompanyFinancialsInput

    def __init__(self, service: CompetitionService | None = None) -> None:
        self._service = service or CompetitionService()

    async def run(self, **kwargs) -> dict:
        inp = CompanyFinancialsInput(**kwargs)
        result = await self._service.get_financials(inp.ticker)
        return dataclasses.asdict(result)
