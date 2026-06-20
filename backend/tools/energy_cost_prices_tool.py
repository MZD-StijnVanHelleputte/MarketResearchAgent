import dataclasses

from pydantic import BaseModel, Field

from services.commodity_service import CommodityService
from tools.base import BaseTool


class EnergyCostPricesInput(BaseModel):
    symbol: str = Field(description="One of WTI, BRENT, or NATURAL_GAS.")
    interval: str = Field(default="monthly", pattern=r"^(daily|weekly|monthly)$")


class EnergyCostPricesTool(BaseTool):
    name = "get_energy_cost_prices"
    description = (
        "Get Alpha Vantage energy-cost series for WTI crude, Brent crude, or natural gas. "
        "Supports daily, weekly, and monthly intervals. Returns symbol, endpoint, interval, "
        "unit, latest observation, rows, and source."
    )
    input_schema = EnergyCostPricesInput

    def __init__(self, service: CommodityService | None = None) -> None:
        self._service = service or CommodityService()

    async def run(self, **kwargs) -> dict:
        inp = EnergyCostPricesInput(**kwargs)
        result = await self._service.get_energy_cost_prices(inp.symbol, inp.interval)
        return dataclasses.asdict(result)
