import dataclasses
from typing import Literal

from pydantic import BaseModel, Field

from services.fred_service import FredService
from tools.base import BaseTool


class FredSeriesUpdatesInput(BaseModel):
    limit: int = Field(default=20, ge=1, le=100)
    filter_value: Literal["macro", "regional", "all"] = Field(
        default="macro",
        description=(
            "macro=national macroeconomic series (default), "
            "regional=state/metro-level series, "
            "all=both."
        ),
    )


class FredSeriesUpdatesTool(BaseTool):
    name = "get_fred_series_updates"
    description = (
        "Get a list of FRED economic series that were recently updated or newly released, "
        "sorted by update timestamp descending. "
        "Useful for tracking which economic data reports were published recently and "
        "discovering fresh releases relevant to the analysis. "
        "filter_value=macro (default) limits to national-level indicators; "
        "filter_value=regional includes state/metro series; filter_value=all returns both."
    )
    input_schema = FredSeriesUpdatesInput

    def __init__(self, service: FredService | None = None) -> None:
        self._service = service or FredService()

    async def run(self, **kwargs) -> dict:
        inp = FredSeriesUpdatesInput(**kwargs)
        results = await self._service.get_series_updates(inp.limit, inp.filter_value)
        return {"results": [dataclasses.asdict(r) for r in results], "count": len(results)}
