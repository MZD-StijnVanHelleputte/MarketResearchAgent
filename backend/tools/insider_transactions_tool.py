import dataclasses

from pydantic import BaseModel, Field

from services.equity_intelligence_service import EquityIntelligenceService
from tools.base import BaseTool


class InsiderTransactionsInput(BaseModel):
    symbol: str = Field(
        description=(
            "Stock ticker of the company to look up insider activity for, e.g. 'CAT', 'DE', 'FCX'. "
            "Note: Alpha Vantage insider data is primarily available for US-listed tickers."
        )
    )


class InsiderTransactionsTool(BaseTool):
    name = "get_insider_transactions"
    requires_premium = "alpha_vantage"
    description = (
        "Get recent insider buying and selling activity from Alpha Vantage INSIDER_TRANSACTIONS. "
        "Signals about executive confidence at competitor companies (CAT, Deere) or major mining "
        "operators (FCX, VALE, RIO). Use for the competition agent or mining_projects agent to "
        "detect unusual insider activity that may foreshadow strategic moves or earnings surprises. "
        "REQUIRES Alpha Vantage premium subscription — raises an error on free-tier keys. "
        "Returns symbol, list of transactions (executive, shares, type, date, price), and source."
    )
    input_schema = InsiderTransactionsInput

    def __init__(self, service: EquityIntelligenceService | None = None) -> None:
        self._service = service or EquityIntelligenceService()

    async def run(self, **kwargs) -> dict:
        inp = InsiderTransactionsInput(**kwargs)
        result = await self._service.get_insider_transactions(inp.symbol)
        return dataclasses.asdict(result)
