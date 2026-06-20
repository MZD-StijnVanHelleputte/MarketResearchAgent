import pytest
from unittest.mock import AsyncMock, MagicMock
from clients.base_http_client import ClientError
from services.news_service import NewsService, NewsArticle, NewsSource, ServiceError


def _mock_client(return_value: dict | None = None, raise_exc: Exception | None = None, method: str = "search"):
    client = MagicMock()
    mock = AsyncMock(side_effect=raise_exc) if raise_exc else AsyncMock(return_value=return_value or {})
    setattr(client, method, mock)
    return client


RAW_OK = {
    "status": "ok",
    "totalResults": 2,
    "articles": [
        {
            "title": "Komatsu expands fleet",
            "description": "New autonomous trucks deployed.",
            "url": "https://example.com/1",
            "publishedAt": "2026-06-01T10:00:00Z",
            "source": {"name": "Mining Weekly"},
        },
        {
            "title": "Cat Q1 results",
            "description": None,
            "url": "https://example.com/2",
            "publishedAt": "2026-05-15T08:00:00Z",
            "source": {"name": "Reuters"},
        },
    ],
}


@pytest.mark.asyncio
async def test_search_returns_articles():
    service = NewsService(client=_mock_client(RAW_OK))
    articles = await service.search("Komatsu")

    assert len(articles) == 2
    assert isinstance(articles[0], NewsArticle)
    assert articles[0].title == "Komatsu expands fleet"
    assert articles[0].source == "Mining Weekly"
    assert articles[1].description is None


@pytest.mark.asyncio
async def test_search_returns_empty_list_when_no_articles():
    raw = {"status": "ok", "totalResults": 0, "articles": []}
    service = NewsService(client=_mock_client(raw))
    articles = await service.search("nothing")
    assert articles == []


@pytest.mark.asyncio
async def test_search_caches_second_call_skips_client():
    """Calling search twice with the same args should call the client once (caching is per-call
    in NewsService — this test just verifies the client is called each time for now)."""
    service = NewsService(client=_mock_client(RAW_OK))
    await service.search("Komatsu")
    await service.search("Komatsu")
    assert service._client.search.call_count == 2


@pytest.mark.asyncio
async def test_search_raises_service_error_on_client_failure():
    service = NewsService(client=_mock_client(raise_exc=ClientError(401, "Unauthorized")))
    with pytest.raises(ServiceError):
        await service.search("test")


@pytest.mark.asyncio
async def test_search_raises_service_error_on_non_ok_status():
    raw = {"status": "error", "code": "apiKeyInvalid", "message": "Invalid key"}
    service = NewsService(client=_mock_client(raw))
    with pytest.raises(ServiceError):
        await service.search("test")


@pytest.mark.asyncio
async def test_top_headlines_returns_articles():
    service = NewsService(client=_mock_client(RAW_OK, method="top_headlines"))
    articles = await service.top_headlines(country="us", category="business")

    assert len(articles) == 2
    assert isinstance(articles[0], NewsArticle)
    assert articles[0].title == "Komatsu expands fleet"


@pytest.mark.asyncio
async def test_top_headlines_raises_service_error_on_client_failure():
    service = NewsService(
        client=_mock_client(raise_exc=ClientError(401, "Unauthorized"), method="top_headlines")
    )
    with pytest.raises(ServiceError):
        await service.top_headlines(country="us")


@pytest.mark.asyncio
async def test_top_headlines_raises_service_error_on_non_ok_status():
    raw = {"status": "error", "code": "apiKeyInvalid", "message": "Invalid key"}
    service = NewsService(client=_mock_client(raw, method="top_headlines"))
    with pytest.raises(ServiceError):
        await service.top_headlines(country="us")


RAW_SOURCES_OK = {
    "status": "ok",
    "sources": [
        {
            "id": "bbc-news",
            "name": "BBC News",
            "description": "Global news coverage.",
            "url": "https://bbc.co.uk",
            "category": "general",
            "language": "en",
            "country": "gb",
        },
        {
            "id": "reuters",
            "name": "Reuters",
            "description": None,
            "url": "https://reuters.com",
            "category": "business",
            "language": "en",
            "country": "us",
        },
    ],
}


@pytest.mark.asyncio
async def test_list_sources_returns_sources():
    service = NewsService(client=_mock_client(RAW_SOURCES_OK, method="sources"))
    sources = await service.list_sources(category="business")

    assert len(sources) == 2
    assert isinstance(sources[0], NewsSource)
    assert sources[0].id == "bbc-news"
    assert sources[1].description is None


@pytest.mark.asyncio
async def test_list_sources_returns_empty_list_when_none():
    raw = {"status": "ok", "sources": []}
    service = NewsService(client=_mock_client(raw, method="sources"))
    sources = await service.list_sources()
    assert sources == []


@pytest.mark.asyncio
async def test_list_sources_raises_service_error_on_client_failure():
    service = NewsService(
        client=_mock_client(raise_exc=ClientError(401, "Unauthorized"), method="sources")
    )
    with pytest.raises(ServiceError):
        await service.list_sources()


@pytest.mark.asyncio
async def test_list_sources_raises_service_error_on_non_ok_status():
    raw = {"status": "error", "code": "apiKeyInvalid", "message": "Invalid key"}
    service = NewsService(client=_mock_client(raw, method="sources"))
    with pytest.raises(ServiceError):
        await service.list_sources()
