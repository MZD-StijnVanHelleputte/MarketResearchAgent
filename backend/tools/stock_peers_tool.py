import dataclasses
from pydantic import BaseModel
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class StockPeersInput(BaseModel):
    ticker: str


class StockPeersTool(BaseTool):
    name = "get_stock_peers"
    requires_premium = "fmp"
    description = (
        "Get the peer companies for a publicly traded company by ticker symbol. "
        "Returns a list of peer ticker symbols in the same sector and market cap range. "
        "Use to discover competitors for benchmarking."
    )
    input_schema = StockPeersInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = StockPeersInput(**kwargs)
        result = await self._service.get_stock_peers(inp.ticker)
        return dataclasses.asdict(result)
