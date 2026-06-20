import dataclasses

from pydantic import BaseModel, Field

from services.fred_service import FredService
from tools.base import BaseTool


class FredReleasesInput(BaseModel):
    limit: int = Field(default=50, ge=1, le=1000)


class FredReleasesTool(BaseTool):
    name = "list_fred_releases"
    description = (
        "List all FRED economic data releases — the named reports that contain series. "
        "Returns release_id, name, and link for each release. "
        "Use the release_id with get_fred_release_series to see the series inside a release. "
        "Notable release IDs: 10=BLS Employment Situation, 11=BLS Producer Price Index, "
        "21=Industrial Production & Capacity Utilization, 53=GDP, 86=CPI, "
        "175=ISM Manufacturing, 184=Durable Goods Orders, 205=Retail Sales. "
        "Useful to discover what economic data reports are available from FRED."
    )
    input_schema = FredReleasesInput

    def __init__(self, service: FredService | None = None) -> None:
        self._service = service or FredService()

    async def run(self, **kwargs) -> dict:
        inp = FredReleasesInput(**kwargs)
        releases = await self._service.list_releases(inp.limit)
        return {"releases": [dataclasses.asdict(r) for r in releases], "count": len(releases)}
