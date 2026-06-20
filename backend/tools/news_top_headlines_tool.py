import dataclasses
from pydantic import BaseModel, Field, model_validator
from tools.base import BaseTool
from services.news_service import NewsService


class NewsTopHeadlinesInput(BaseModel):
    query: str | None = None
    country: str | None = None
    category: str | None = None  # business, entertainment, general, health, science, sports, technology
    sources: str | None = None  # comma-separated NewsAPI source ids
    page_size: int = Field(default=5, ge=1, le=20)
    page: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _check_sources_exclusivity(self):
        if self.sources and (self.country or self.category):
            raise ValueError(
                "NewsAPI does not allow combining 'sources' with 'country' or 'category'."
            )
        return self


class NewsTopHeadlinesTool(BaseTool):
    name = "news_top_headlines"
    description = (
        "Get breaking/top news headlines, optionally filtered by country, category, "
        "or specific source IDs (use news_sources to discover valid source IDs). "
        "'sources' cannot be combined with 'country' or 'category'. "
        "Use for breaking-news monitoring; use news_search for full-text historical search."
    )
    input_schema = NewsTopHeadlinesInput

    def __init__(self, service: NewsService | None = None) -> None:
        self._service = service or NewsService()

    async def run(self, **kwargs) -> dict:
        inp = NewsTopHeadlinesInput(**kwargs)
        articles = await self._service.top_headlines(
            query=inp.query,
            country=inp.country,
            category=inp.category,
            sources=inp.sources,
            page_size=inp.page_size,
            page=inp.page,
        )
        return {"articles": [dataclasses.asdict(a) for a in articles]}
