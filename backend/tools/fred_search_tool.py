import dataclasses

from pydantic import BaseModel, Field

from services.fred_service import FredService
from tools.base import BaseTool


class FredSearchInput(BaseModel):
    search_text: str
    limit: int = Field(default=10, ge=1, le=100)


class FredSearchTool(BaseTool):
    name = "search_fred_series"
    description = (
        "Search the full FRED (Federal Reserve Economic Data) catalogue by keyword to discover "
        "relevant economic series when you don't know the exact series_id. "
        "Returns matching series with their id, title, units, frequency, and popularity score. "
        "Example searches: 'steel production', 'construction spending', 'manufacturing PMI', "
        "'crude oil imports', 'China trade', 'copper price', 'truck sales'. "
        "Use the returned series_id with get_fred_observations or get_macro_indicator to fetch data."
    )
    input_schema = FredSearchInput

    def __init__(self, service: FredService | None = None) -> None:
        self._service = service or FredService()

    async def run(self, **kwargs) -> dict:
        inp = FredSearchInput(**kwargs)
        results = await self._service.search_series(inp.search_text, inp.limit)
        return {"results": [dataclasses.asdict(r) for r in results], "count": len(results)}
