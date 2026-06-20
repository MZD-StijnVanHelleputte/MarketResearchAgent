import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.equity_service import EquityService


class EquityHistoryInput(BaseModel):
    ticker: str = Field(description="Stock ticker symbol, e.g. 'CAT', 'VOLV-B.ST', '6305.T'.")
    period: str = Field(
        default="1y",
        description=(
            "Lookback period for daily OHLCV data. "
            "Options: 1mo, 3mo, 6mo, 1y, 2y, 5y. "
            "Default is 1y (one full year). Use 2y or 5y for long-term trend analysis."
        ),
    )


class EquityHistoryTool(BaseTool):
    name = "get_equity_history"
    description = (
        "Get daily OHLCV (open/high/low/close/volume) price history for a stock ticker "
        "over a configurable period (default 1y, up to 5y). Use to analyse year-over-year "
        "price performance, volatility, drawdowns, and multi-year trends for competitors "
        "or major mining operators. Works for major exchanges (NYSE, NASDAQ, TSE, LSE, etc.). "
        "Returns ticker, period, and a list of daily rows with date, open, high, low, close, volume."
    )
    input_schema = EquityHistoryInput

    def __init__(self, service: EquityService | None = None) -> None:
        self._service = service or EquityService()

    async def run(self, **kwargs) -> dict:
        inp = EquityHistoryInput(**kwargs)
        result = await self._service.get_history(inp.ticker, inp.period)
        return dataclasses.asdict(result)
