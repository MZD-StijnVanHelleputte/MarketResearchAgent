import pytest
from unittest.mock import AsyncMock, MagicMock
from clients.base_http_client import ClientError
from services.web_search_service import WebSearchService, SearchResult, ServiceError

SEARCH_RAW = {
    "results": [
        {"title": "Komatsu Mining", "url": "https://example.com", "content": "Mining equipment leader", "score": 0.92},
        {"title": "Komatsu News", "url": "https://example.com/news", "content": "Latest updates", "score": 0.75},
    ]
}


def _mock_client(raw=SEARCH_RAW):
    client = MagicMock()
    client.search = AsyncMock(return_value=raw)
    return client


@pytest.mark.asyncio
async def test_search_returns_results():
    svc = WebSearchService(client=_mock_client())
    results = await svc.search("Komatsu mining equipment")
    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].title == "Komatsu Mining"
    assert results[0].score == 0.92


@pytest.mark.asyncio
async def test_search_empty_results():
    svc = WebSearchService(client=_mock_client({"results": []}))
    results = await svc.search("nothing")
    assert results == []


@pytest.mark.asyncio
async def test_search_raises_service_error_on_client_error():
    client = MagicMock()
    client.search = AsyncMock(side_effect=ClientError(401, "Unauthorized"))
    svc = WebSearchService(client=client)
    with pytest.raises(ServiceError):
        await svc.search("test")
