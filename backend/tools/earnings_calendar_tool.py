import dataclasses

from pydantic import BaseModel, Field

from services.equity_intelligence_service import EquityIntelligenceService
from tools.base import BaseTool


class EarningsCalendarInput(BaseModel):
    symbol: str = Field(
        default="",
        description=(
            "Stock ticker to filter by, e.g. 'CAT', 'SAND', 'EPIR'. "
            "Leave empty to retrieve upcoming earnings across the broad market."
        ),
    )
    horizon: str = Field(
        default="3month",
        pattern=r"^(3month|6month|12month)$",
        description="Lookahead window: 3month, 6month, or 12month.",
    )


class EarningsCalendarTool(BaseTool):
    name = "get_earnings_calendar"
    description = (
        "Get upcoming earnings announcement dates from Alpha Vantage EARNINGS_CALENDAR. "
        "Use to know when competitor companies (CAT, Volvo CE, Sandvik, Epiroc, Komatsu) "
        "will report, so the competition agent can time deeper analysis around those events. "
        "Returns symbol, horizon, list of events (symbol, name, report_date, estimate, currency), "
        "and source. Free tier — no premium subscription required."
    )
    input_schema = EarningsCalendarInput

    def __init__(self, service: EquityIntelligenceService | None = None) -> None:
        self._service = service or EquityIntelligenceService()

    async def run(self, **kwargs) -> dict:
        inp = EarningsCalendarInput(**kwargs)
        result = await self._service.get_earnings_calendar(inp.symbol, inp.horizon)
        return dataclasses.asdict(result)
