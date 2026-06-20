import dataclasses

from pydantic import BaseModel, Field

from services.macro_service import MacroService
from tools.base import BaseTool


class MacroIndicatorsInput(BaseModel):
    series_id: str
    limit: int = Field(default=10, ge=1, le=50)


class MacroIndicatorsTool(BaseTool):
    name = "get_macro_indicator"
    description = (
        "Fetch a macroeconomic time series from FRED (Federal Reserve Economic Data). "
        "Returns the most recent observations with title and units. "
        "Useful series_ids: FEDFUNDS (Fed funds rate), GDP (US GDP), CPIAUCSL (CPI inflation), "
        "INDPRO (industrial production index), DGS10 (10-year Treasury rate), "
        "HOUST (housing starts), DCOILWTICO (crude oil price WTI), "
        "DEXJPUS (JPY/USD exchange rate), DEXCHUS (CNY/USD exchange rate), "
        "UNRATE (US unemployment rate)."
    )
    input_schema = MacroIndicatorsInput

    def __init__(self, service: MacroService | None = None) -> None:
        self._service = service or MacroService()

    async def run(self, **kwargs) -> dict:
        inp = MacroIndicatorsInput(**kwargs)
        indicator = await self._service.get_indicator(inp.series_id, inp.limit)
        return dataclasses.asdict(indicator)
