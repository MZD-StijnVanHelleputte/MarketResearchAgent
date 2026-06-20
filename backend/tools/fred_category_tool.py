from pydantic import BaseModel, Field

from services.fred_service import FredService
from tools.base import BaseTool


class FredCategoryInput(BaseModel):
    category_id: int = Field(default=0, description="FRED category ID. Use 0 for the root.")
    include_series: bool = Field(
        default=False,
        description="If true, also return the most popular series within this category.",
    )
    series_limit: int = Field(default=20, ge=1, le=100)


class FredCategoryTool(BaseTool):
    name = "browse_fred_category"
    description = (
        "Browse the FRED category hierarchy to discover series by economic topic. "
        "Returns the category name, its child categories (with IDs), and optionally "
        "the most popular series within the category. "
        "Start with category_id=0 (root) to see top-level topics, then drill into children. "
        "Key category IDs: 0=root, 32991=Money Banking&Finance, 32992=Population/Employment/Labor, "
        "10=Business Cycles, 32262=Production & Business Activity, 32455=Prices, "
        "33936=Trade & International Transactions, 3008=Wholesale Trade, "
        "97=Business/Consumer Surveys. "
        "Set include_series=true to see actual series_ids once you've found the right category."
    )
    input_schema = FredCategoryInput

    def __init__(self, service: FredService | None = None) -> None:
        self._service = service or FredService()

    async def run(self, **kwargs) -> dict:
        inp = FredCategoryInput(**kwargs)
        return await self._service.browse_category(
            category_id=inp.category_id,
            include_series=inp.include_series,
            series_limit=inp.series_limit,
        )
