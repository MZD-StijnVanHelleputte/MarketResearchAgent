import pytest
from unittest.mock import AsyncMock, patch
from clients.newsapi_client import NewsApiClient


@pytest.mark.asyncio
async def test_search_forwards_params():
    client = NewsApiClient()
    raw_response = {
        "status": "ok",
        "totalResults": 1,
        "articles": [{"title": "Test", "url": "https://example.com"}],
    }

    with patch.object(client, "get", new=AsyncMock(return_value=raw_response)) as mock_get:
        result = await client.search(query="Komatsu mining", page_size=3, from_date="2026-01-01")

    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert call_args[0][0] == "/v2/everything"
    params = call_args[1]["params"]
    assert params["q"] == "Komatsu mining"
    assert params["pageSize"] == 3
    assert params["from"] == "2026-01-01"
    assert "apiKey" in params
    assert result == raw_response


@pytest.mark.asyncio
async def test_search_omits_from_date_when_none():
    client = NewsApiClient()
    with patch.object(client, "get", new=AsyncMock(return_value={"status": "ok", "articles": []})) as mock_get:
        await client.search(query="copper")

    params = mock_get.call_args[1]["params"]
    assert "from" not in params


@pytest.mark.asyncio
async def test_top_headlines_forwards_params():
    client = NewsApiClient()
    raw_response = {
        "status": "ok",
        "totalResults": 1,
        "articles": [{"title": "Breaking"}],
    }

    with patch.object(client, "get", new=AsyncMock(return_value=raw_response)) as mock_get:
        result = await client.top_headlines(country="us", category="business", page_size=10)

    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert call_args[0][0] == "/v2/top-headlines"
    params = call_args[1]["params"]
    assert params["country"] == "us"
    assert params["category"] == "business"
    assert params["pageSize"] == 10
    assert "apiKey" in params
    assert result == raw_response


@pytest.mark.asyncio
async def test_top_headlines_omits_unset_filters():
    client = NewsApiClient()
    with patch.object(client, "get", new=AsyncMock(return_value={"status": "ok", "articles": []})) as mock_get:
        await client.top_headlines()

    params = mock_get.call_args[1]["params"]
    assert "country" not in params
    assert "category" not in params
    assert "sources" not in params
    assert "q" not in params


@pytest.mark.asyncio
async def test_sources_forwards_params():
    client = NewsApiClient()
    raw_response = {"status": "ok", "sources": [{"id": "bbc-news", "name": "BBC News"}]}

    with patch.object(client, "get", new=AsyncMock(return_value=raw_response)) as mock_get:
        result = await client.sources(category="technology", language="en", country="gb")

    mock_get.assert_called_once()
    call_args = mock_get.call_args
    assert call_args[0][0] == "/v2/top-headlines/sources"
    params = call_args[1]["params"]
    assert params["category"] == "technology"
    assert params["language"] == "en"
    assert params["country"] == "gb"
    assert result == raw_response


@pytest.mark.asyncio
async def test_sources_omits_unset_filters():
    client = NewsApiClient()
    with patch.object(client, "get", new=AsyncMock(return_value={"status": "ok", "sources": []})) as mock_get:
        await client.sources()

    params = mock_get.call_args[1]["params"]
    assert "category" not in params
    assert "language" not in params
    assert "country" not in params
