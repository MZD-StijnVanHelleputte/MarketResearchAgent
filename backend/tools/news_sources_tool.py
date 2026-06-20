import dataclasses
from pydantic import BaseModel
from tools.base import BaseTool
from services.news_service import NewsService


class NewsSourcesInput(BaseModel):
    category: str | None = None  # business, entertainment, general, health, science, sports, technology
    language: str | None = None
    country: str | None = None


class NewsSourcesTool(BaseTool):
    name = "news_sources"
    description = (
        "List available NewsAPI news sources/publishers, optionally filtered by "
        "category, language, or country. Returns source id, name, description, "
        "url, category, language, country. Use the returned 'id' values as the "
        "'sources' parameter for news_top_headlines or news_search."
    )
    input_schema = NewsSourcesInput

    def __init__(self, service: NewsService | None = None) -> None:
        self._service = service or NewsService()

    async def run(self, **kwargs) -> dict:
        inp = NewsSourcesInput(**kwargs)
        sources = await self._service.list_sources(
            category=inp.category,
            language=inp.language,
            country=inp.country,
        )
        return {"sources": [dataclasses.asdict(s) for s in sources]}
