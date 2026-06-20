import dataclasses

from pydantic import BaseModel, Field

from services.commodity_service import CommodityService
from tools.base import BaseTool


class MiningMetalsPricesInput(BaseModel):
    symbol: str = Field(description="One of COPPER, ALUMINUM, GOLD, SILVER, XAU, or XAG.")
    interval: str = Field(default="monthly", description="daily, weekly, monthly, quarterly, or annual depending on symbol.")
    include_history: bool = Field(default=True, description="For GOLD/SILVER, use history when true and live spot when false.")


class MiningMetalsPricesTool(BaseTool):
    name = "get_mining_metals_prices"
    description = (
        "Get Alpha Vantage mining metal price data for COPPER, ALUMINUM, GOLD, or SILVER. "
        "Copper and aluminum use their commodity series. Gold and silver use the dedicated "
        "GOLD_SILVER_HISTORY endpoint by default, or GOLD_SILVER_SPOT when include_history=false. "
        "Returns symbol, endpoint, interval, unit, latest observation, rows, and source."
    )
    input_schema = MiningMetalsPricesInput

    def __init__(self, service: CommodityService | None = None) -> None:
        self._service = service or CommodityService()

    async def run(self, **kwargs) -> dict:
        inp = MiningMetalsPricesInput(**kwargs)
        result = await self._service.get_mining_metals_prices(
            inp.symbol,
            inp.interval,
            inp.include_history,
        )
        return dataclasses.asdict(result)
