import dataclasses

from pydantic import BaseModel, Field

from services.equity_intelligence_service import EquityIntelligenceService
from tools.base import BaseTool


class NewsSentimentInput(BaseModel):
    tickers: str = Field(
        default="",
        description=(
            "Comma-separated ticker symbols to filter articles by, e.g. 'CAT,VOLV-B.ST,EPIR.ST'. "
            "Leave empty to retrieve across all tickers."
        ),
    )
    topics: str = Field(
        default="",
        description=(
            "Comma-separated Alpha Vantage topic filters, e.g. 'mining,construction,earnings'. "
            "Available: earnings, ipo, mergers_and_acquisitions, financial_markets, economy_fiscal, "
            "economy_monetary, economy_macro, energy_transportation, finance, life_sciences, "
            "manufacturing, real_estate, retail_wholesale, technology."
        ),
    )
    sort: str = Field(
        default="LATEST",
        pattern=r"^(LATEST|RELEVANCE|SENTIMENT)$",
        description="Sort order: LATEST, RELEVANCE, or SENTIMENT.",
    )
    limit: int = Field(
        default=25,
        ge=1,
        le=50,
        description="Number of articles to return (1–50).",
    )


class NewsSentimentTool(BaseTool):
    name = "get_news_sentiment"
    requires_premium = "alpha_vantage"
    description = (
        "Get sentiment-scored news articles from Alpha Vantage NEWS_SENTIMENT (Alpha Intelligence). "
        "Returns articles with overall sentiment score/label and per-ticker sentiment breakdowns. "
        "Use for the competition agent (filter by CAT, SAND, VOLV-B.ST) or mining_projects agent "
        "(filter topics='mining'). Complements news_search (NewsAPI) with quantified sentiment. "
        "REQUIRES Alpha Vantage premium subscription — raises an error on free-tier keys. "
        "Returns tickers, topics, sort, items_fetched, articles list, and source."
    )
    input_schema = NewsSentimentInput

    def __init__(self, service: EquityIntelligenceService | None = None) -> None:
        self._service = service or EquityIntelligenceService()

    async def run(self, **kwargs) -> dict:
        inp = NewsSentimentInput(**kwargs)
        result = await self._service.get_news_sentiment(
            inp.tickers, inp.topics, inp.sort, inp.limit
        )
        return dataclasses.asdict(result)
