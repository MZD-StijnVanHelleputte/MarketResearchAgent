import dataclasses

from pydantic import BaseModel, Field

from services.commodity_service import CommodityService
from tools.base import BaseTool


class BroadCommodityCycleInput(BaseModel):
    interval: str = Field(default="monthly", pattern=r"^(monthly|quarterly|annual)$")


class BroadCommodityCycleTool(BaseTool):
    name = "get_broad_commodity_cycle"
    description = (
        "Get Alpha Vantage ALL_COMMODITIES broad commodity price index data. "
        "Supports monthly, quarterly, and annual intervals. Returns symbol, endpoint, "
        "interval, unit, latest observation, rows, and source."
    )
    input_schema = BroadCommodityCycleInput

    def __init__(self, service: CommodityService | None = None) -> None:
        self._service = service or CommodityService()

    async def run(self, **kwargs) -> dict:
        inp = BroadCommodityCycleInput(**kwargs)
        result = await self._service.get_broad_commodity_cycle(inp.interval)
        return dataclasses.asdict(result)
