import dataclasses

from pydantic import BaseModel, Field

from services.equity_intelligence_service import EquityIntelligenceService
from tools.base import BaseTool


class EarningsTranscriptInput(BaseModel):
    symbol: str = Field(
        description=(
            "Stock ticker of the company whose earnings call you want, e.g. 'CAT', 'DE', 'EPIR.ST'. "
            "Note: Alpha Vantage transcript coverage is primarily for US-listed tickers."
        )
    )
    quarter: str = Field(
        description=(
            "Fiscal quarter in the format YYYYQn, e.g. '2024Q4' or '2025Q1'. "
            "Use the most recent completed quarter for the latest transcript."
        )
    )


class EarningsTranscriptTool(BaseTool):
    name = "get_earnings_transcript"
    requires_premium = "alpha_vantage"
    description = (
        "Retrieve a parsed earnings call transcript from Alpha Vantage EARNINGS_CALL_TRANSCRIPT. "
        "Returns speaker-level segments and raw transcript text. Use for the competition agent "
        "to extract what executives at Caterpillar, Deere, or other competitors said about "
        "equipment demand, mining markets, or product pipeline. "
        "REQUIRES Alpha Vantage premium subscription — raises an error on free-tier keys. "
        "Returns symbol, quarter, segments (speaker + text), raw_text, and source."
    )
    input_schema = EarningsTranscriptInput

    def __init__(self, service: EquityIntelligenceService | None = None) -> None:
        self._service = service or EquityIntelligenceService()

    async def run(self, **kwargs) -> dict:
        inp = EarningsTranscriptInput(**kwargs)
        result = await self._service.get_earnings_transcript(inp.symbol, inp.quarter)
        return dataclasses.asdict(result)
