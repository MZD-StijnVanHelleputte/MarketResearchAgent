import pytest
from unittest.mock import AsyncMock, MagicMock
from pydantic import ValidationError
from services.news_service import NewsArticle
from tools.news_top_headlines_tool import NewsTopHeadlinesTool, NewsTopHeadlinesInput


def _mock_service(articles: list[NewsArticle]):
    svc = MagicMock()
    svc.top_headlines = AsyncMock(return_value=articles)
    return svc


ARTICLES = [
    NewsArticle(
        title="Komatsu launches new excavator",
        description="Details here.",
        url="https://example.com/1",
        published_at="2026-06-01T10:00:00Z",
        source="Mining Weekly",
    )
]


@pytest.mark.asyncio
async def test_run_returns_articles_dict():
    tool = NewsTopHeadlinesTool(service=_mock_service(ARTICLES))
    result = await tool.run(country="us", category="business")

    assert "articles" in result
    assert len(result["articles"]) == 1
    assert result["articles"][0]["title"] == "Komatsu launches new excavator"


@pytest.mark.asyncio
async def test_run_passes_all_args_to_service():
    svc = _mock_service([])
    tool = NewsTopHeadlinesTool(service=svc)
    await tool.run(query="copper", country="us", category="business", page_size=10, page=2)

    svc.top_headlines.assert_called_once_with(
        query="copper",
        country="us",
        category="business",
        sources=None,
        page_size=10,
        page=2,
    )


@pytest.mark.asyncio
async def test_run_empty_results():
    tool = NewsTopHeadlinesTool(service=_mock_service([]))
    result = await tool.run(country="us")
    assert result == {"articles": []}


def test_sources_with_country_raises_validation_error():
    with pytest.raises(ValidationError):
        NewsTopHeadlinesInput(sources="bbc-news", country="us")


def test_sources_with_category_raises_validation_error():
    with pytest.raises(ValidationError):
        NewsTopHeadlinesInput(sources="bbc-news", category="business")


def test_sources_alone_is_valid():
    inp = NewsTopHeadlinesInput(sources="bbc-news")
    assert inp.sources == "bbc-news"


def test_tool_metadata():
    tool = NewsTopHeadlinesTool()
    assert tool.name == "news_top_headlines"
    assert "headlines" in tool.description.lower()
