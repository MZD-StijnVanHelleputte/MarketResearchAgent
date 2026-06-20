from pydantic import BaseModel, Field

from services.fred_service import FredService
from tools.base import BaseTool


class FredReleaseSeriesInput(BaseModel):
    release_id: int
    limit: int = Field(default=20, ge=1, le=100)


class FredReleaseSeriesToolCls(BaseTool):
    name = "get_fred_release_series"
    description = (
        "Get the economic data series that belong to a specific FRED release, sorted by popularity. "
        "Also returns the release name and link. "
        "Use list_fred_releases to find release IDs. "
        "Notable release IDs: 10=BLS Employment Situation (UNRATE, PAYEMS, etc.), "
        "11=PPI, 21=Industrial Production (INDPRO), 53=GDP, 86=CPI (CPIAUCSL), "
        "175=ISM Manufacturing, 184=Durable Goods. "
        "Useful when you know which report to pull from but need to discover its series_ids."
    )
    input_schema = FredReleaseSeriesInput

    def __init__(self, service: FredService | None = None) -> None:
        self._service = service or FredService()

    async def run(self, **kwargs) -> dict:
        inp = FredReleaseSeriesInput(**kwargs)
        return await self._service.get_release_with_series(inp.release_id, inp.limit)
