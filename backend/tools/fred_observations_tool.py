import dataclasses
from typing import Literal

from pydantic import BaseModel, Field

from services.fred_service import FredService
from tools.base import BaseTool


class FredObservationsInput(BaseModel):
    series_id: str
    observation_start: str | None = Field(
        default=None,
        description="Start date in YYYY-MM-DD format. Omit to start from earliest available.",
    )
    observation_end: str | None = Field(
        default=None,
        description="End date in YYYY-MM-DD format. Omit to include up to most recent.",
    )
    limit: int = Field(default=100, ge=1, le=100000)
    units: Literal["lin", "chg", "ch1", "pch", "pc1", "pca", "cch", "cca", "log"] = "lin"
    frequency: str | None = Field(
        default=None,
        description=(
            "Aggregate to a lower frequency: d=daily, w=weekly, bw=biweekly, "
            "m=monthly, q=quarterly, sa=semiannual, a=annual. "
            "Omit to use the native frequency of the series."
        ),
    )


class FredObservationsTool(BaseTool):
    name = "get_fred_observations"
    description = (
        "Fetch FRED economic time-series observations with full control over date range, "
        "frequency aggregation, and unit transformation. "
        "Prefer this over get_macro_indicator when you need: a specific date range "
        "(e.g. 2018-01-01 to 2023-12-31), percent-change units (units='pch' or 'pc1'), "
        "or frequency downsampling (e.g. daily data aggregated to monthly). "
        "units options: lin=levels, chg=change, ch1=change from year ago, "
        "pch=percent change, pc1=percent change from year ago, pca=compounded annual rate, "
        "log=natural log. "
        "Returns series metadata plus the observations list."
    )
    input_schema = FredObservationsInput

    def __init__(self, service: FredService | None = None) -> None:
        self._service = service or FredService()

    async def run(self, **kwargs) -> dict:
        inp = FredObservationsInput(**kwargs)
        result = await self._service.get_observations(
            series_id=inp.series_id,
            observation_start=inp.observation_start,
            observation_end=inp.observation_end,
            limit=inp.limit,
            units=inp.units,
            frequency=inp.frequency,
        )
        return dataclasses.asdict(result)
