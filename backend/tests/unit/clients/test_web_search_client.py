import pytest
from unittest.mock import AsyncMock, patch
from clients.web_search_client import WebSearchClient
from clients.base_http_client import ClientError

SEARCH_RESPONSE = {
    "results": [
        {"title": "Komatsu Mining", "url": "https://example.com", "content": "snippet", "score": 0.9}
    ]
}


@pytest.mark.asyncio
async def test_search_forwards_query():
    client = WebSearchClient()
    with patch.object(client, "post", new=AsyncMock(return_value=SEARCH_RESPONSE)) as mock_post:
        result = await client.search("Komatsu mining equipment", max_results=3)

    mock_post.assert_called_once()
    body = mock_post.call_args[1]["json"]
    assert body["query"] == "Komatsu mining equipment"
    assert body["max_results"] == 3
    assert "api_key" in body
    assert result == SEARCH_RESPONSE


def test_auth_headers_empty():
    client = WebSearchClient()
    assert client._auth_headers() == {}


@pytest.mark.asyncio
async def test_raises_client_error_on_4xx():
    client = WebSearchClient()
    with patch.object(client, "post", new=AsyncMock(side_effect=ClientError(401, "Unauthorized"))):
        with pytest.raises(ClientError) as exc_info:
            await client.search("test")
    assert exc_info.value.status == 401
