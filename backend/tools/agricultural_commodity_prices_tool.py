import dataclasses

from pydantic import BaseModel, Field

from services.commodity_service import CommodityService
from tools.base import BaseTool


class AgriculturalCommodityPricesInput(BaseModel):
    symbol: str = Field(description="One of WHEAT or CORN.")
    interval: str = Field(
        default="monthly",
        pattern=r"^(monthly|quarterly|annual)$",
        description="monthly, quarterly, or annual.",
    )


class AgriculturalCommodityPricesTool(BaseTool):
    name = "get_agricultural_commodity_prices"
    description = (
        "Get Alpha Vantage agricultural commodity price series for WHEAT or CORN. "
        "Useful as a food-inflation and emerging-market macro signal for mining regions "
        "(Africa, South America) where food costs affect project economics and social risk. "
        "Supports monthly, quarterly, and annual intervals. "
        "Returns symbol, endpoint, interval, unit, latest observation, rows, and source."
    )
    input_schema = AgriculturalCommodityPricesInput

    def __init__(self, service: CommodityService | None = None) -> None:
        self._service = service or CommodityService()

    async def run(self, **kwargs) -> dict:
        inp = AgriculturalCommodityPricesInput(**kwargs)
        result = await self._service.get_agricultural_commodity_prices(inp.symbol, inp.interval)
        return dataclasses.asdict(result)
