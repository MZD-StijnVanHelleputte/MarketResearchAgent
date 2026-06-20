import dataclasses

from pydantic import BaseModel, Field

from services.fred_service import FredService
from tools.base import BaseTool


class FredTagsSeriesInput(BaseModel):
    tag_names: str = Field(
        description=(
            "Semicolon-separated FRED tag names. The result contains series matching ALL tags. "
            "Examples: 'manufacturing;monthly;sa' or 'construction;quarterly' or 'copper;annual'. "
            "Common tags: nsa=not seasonally adjusted, sa=seasonally adjusted, "
            "monthly, quarterly, annual, daily, weekly, index, rate, price, "
            "manufacturing, construction, trade, employment, price, gdp."
        )
    )
    limit: int = Field(default=20, ge=1, le=100)


class FredTagsSeriesTool(BaseTool):
    name = "get_fred_series_by_tags"
    description = (
        "Find FRED economic series that match ALL of a set of topic tags. "
        "Returns series sorted by popularity. Use tag_names to narrow by topic, frequency, "
        "and seasonal adjustment simultaneously. "
        "Semicolon-separate multiple tags — e.g. 'manufacturing;monthly;sa' returns monthly, "
        "seasonally-adjusted manufacturing series. "
        "Common tags: manufacturing, construction, trade, employment, price, gdp, index, rate; "
        "frequencies: daily, weekly, monthly, quarterly, annual; "
        "adjustment: sa (seasonally adjusted), nsa (not seasonally adjusted). "
        "Use search_fred_series for free-text search; use this tool for structured tag filtering."
    )
    input_schema = FredTagsSeriesInput

    def __init__(self, service: FredService | None = None) -> None:
        self._service = service or FredService()

    async def run(self, **kwargs) -> dict:
        inp = FredTagsSeriesInput(**kwargs)
        results = await self._service.get_series_by_tags(inp.tag_names, inp.limit)
        return {"results": [dataclasses.asdict(r) for r in results], "count": len(results)}
