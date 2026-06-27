from datetime import date, timedelta
import pytest
from unittest.mock import AsyncMock, MagicMock
from config.settings import settings
from services.news_service import NewsArticle
from tools.news_search_tool import NewsSearchTool


def _mock_service(articles: list[NewsArticle]):
    svc = MagicMock()
    svc.search = AsyncMock(return_value=articles)
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
    tool = NewsSearchTool(service=_mock_service(ARTICLES))
    result = await tool.run(query="Komatsu excavator")

    assert "articles" in result
    assert len(result["articles"]) == 1
    assert result["articles"][0]["title"] == "Komatsu launches new excavator"
    assert result["articles"][0]["source"] == "Mining Weekly"


@pytest.mark.asyncio
async def test_run_passes_all_args_to_service():
    svc = _mock_service([])
    tool = NewsSearchTool(service=svc)
    recent_date = (date.today() - timedelta(days=5)).isoformat()
    await tool.run(query="copper", language="fr", page_size=10, from_date=recent_date)

    svc.search.assert_called_once_with(
        query="copper",
        language="fr",
        page_size=10,
        from_date=recent_date,
    )


@pytest.mark.asyncio
async def test_run_clamps_from_date_on_free_tier(monkeypatch):
    monkeypatch.setattr(settings, "newsapi_tier", "free")
    svc = _mock_service([])
    tool = NewsSearchTool(service=svc)
    far_past = (date.today() - timedelta(days=365)).isoformat()
    await tool.run(query="copper", from_date=far_past)

    called_from = svc.search.call_args.kwargs["from_date"]
    expected_earliest = (date.today() - timedelta(days=30)).isoformat()
    assert called_from == expected_earliest


@pytest.mark.asyncio
async def test_run_allows_long_lookback_on_premium_tier(monkeypatch):
    monkeypatch.setattr(settings, "newsapi_tier", "premium")
    svc = _mock_service([])
    tool = NewsSearchTool(service=svc)
    far_past = (date.today() - timedelta(days=365)).isoformat()
    await tool.run(query="copper", from_date=far_past)

    called_from = svc.search.call_args.kwargs["from_date"]
    assert called_from == far_past


@pytest.mark.asyncio
async def test_run_empty_results():
    tool = NewsSearchTool(service=_mock_service([]))
    result = await tool.run(query="nothing")
    assert result == {"articles": []}


def test_tool_metadata():
    tool = NewsSearchTool()
    assert tool.name == "news_search"
    assert "news" in tool.description.lower()
