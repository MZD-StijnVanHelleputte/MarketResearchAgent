import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.fundamentals_service import FundamentalsService


class PressReleasesInput(BaseModel):
    ticker: str
    limit: int = Field(default=10, ge=1, le=50)


class PressReleasesTool(BaseTool):
    name = "get_press_releases"
    requires_premium = "fmp"
    description = (
        "Get recent official press releases for a publicly traded company. "
        "Returns titles, dates, and text content (up to 2000 chars each). "
        "Use for M&A announcements, strategic partnerships, product launches, and guidance updates."
    )
    input_schema = PressReleasesInput

    def __init__(self, service: FundamentalsService | None = None) -> None:
        self._service = service or FundamentalsService()

    async def run(self, **kwargs) -> dict:
        inp = PressReleasesInput(**kwargs)
        results = await self._service.get_press_releases(inp.ticker, limit=inp.limit)
        return {"press_releases": [dataclasses.asdict(r) for r in results]}
