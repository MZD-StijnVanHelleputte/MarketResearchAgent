import dataclasses
from pydantic import BaseModel
from tools.base import BaseTool
from services.equity_service import EquityService


class EquityPriceInput(BaseModel):
    ticker: str  # e.g. "CAT", "SAND.ST", "6305.T"


class EquityPriceTool(BaseTool):
    name = "get_equity_price"
    description = (
        "Get the latest market price, currency, and market cap for a stock ticker. "
        "Works for major exchanges (NYSE, NASDAQ, TSE, LSE, etc.)."
    )
    input_schema = EquityPriceInput

    def __init__(self, service: EquityService | None = None) -> None:
        self._service = service or EquityService()

    async def run(self, **kwargs) -> dict:
        inp = EquityPriceInput(**kwargs)
        result = await self._service.get_price(inp.ticker)
        return dataclasses.asdict(result)
