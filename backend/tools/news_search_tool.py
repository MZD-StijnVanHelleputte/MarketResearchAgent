import dataclasses
from datetime import date, timedelta
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.news_service import NewsService
from config.settings import settings


class NewsSearchInput(BaseModel):
    query: str
    language: str = "en"
    page_size: int = Field(default=5, ge=1, le=20)
    from_date: str | None = None  # ISO date YYYY-MM-DD


class NewsSearchTool(BaseTool):
    name = "news_search"
    description = (
        "Search recent news articles by keyword. "
        "Returns titles, descriptions, URLs, and publication dates."
    )
    input_schema = NewsSearchInput

    def __init__(self, service: NewsService | None = None) -> None:
        self._service = service or NewsService()

    def _max_lookback_days(self) -> int:
        if settings.newsapi_tier == "premium":
            return settings.newsapi_premium_max_lookback_days
        return settings.newsapi_free_max_lookback_days

    async def run(self, **kwargs) -> dict:
        inp = NewsSearchInput(**kwargs)
        max_lookback_days = self._max_lookback_days()
        earliest_allowed = date.today() - timedelta(days=max_lookback_days)
        default_from = date.today() - timedelta(days=min(30, max_lookback_days))

        if inp.from_date:
            requested_from = date.fromisoformat(inp.from_date)
            effective_from = max(requested_from, earliest_allowed).isoformat()
        else:
            effective_from = default_from.isoformat()

        articles = await self._service.search(
            query=inp.query,
            language=inp.language,
            page_size=inp.page_size,
            from_date=effective_from,
        )
        return {"articles": [dataclasses.asdict(a) for a in articles]}
