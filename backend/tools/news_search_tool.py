import dataclasses
from datetime import date, timedelta
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.news_service import NewsService


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

    async def run(self, **kwargs) -> dict:
        inp = NewsSearchInput(**kwargs)
        effective_from = inp.from_date or (date.today() - timedelta(days=30)).isoformat()
        articles = await self._service.search(
            query=inp.query,
            language=inp.language,
            page_size=inp.page_size,
            from_date=effective_from,
        )
        return {"articles": [dataclasses.asdict(a) for a in articles]}
