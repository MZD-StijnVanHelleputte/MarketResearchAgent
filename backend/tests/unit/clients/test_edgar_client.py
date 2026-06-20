import pytest
from unittest.mock import AsyncMock, patch
from clients.edgar_client import EdgarClient
from clients.base_http_client import ClientError

SEARCH_RESPONSE = {
    "hits": {
        "hits": [
            {"_source": {"entity_name": "Caterpillar Inc", "form_type": "10-K", "file_date": "2024-02-14", "period_of_report": "2023-12-31"}}
        ]
    }
}


@pytest.mark.asyncio
async def test_search_filings_returns_dict():
    client = EdgarClient()
    with patch.object(client, "get", new=AsyncMock(return_value=SEARCH_RESPONSE)) as mock_get:
        result = await client.search_filings("Caterpillar mining")

    mock_get.assert_called_once()
    assert result == SEARCH_RESPONSE


@pytest.mark.asyncio
async def test_search_filings_passes_correct_params():
    client = EdgarClient()
    with patch.object(client, "get", new=AsyncMock(return_value={})) as mock_get:
        await client.search_filings("Komatsu", forms="10-K", limit=3)

    path = mock_get.call_args[0][0]
    params = mock_get.call_args[1]["params"]
    assert path == "/LATEST/search-index"
    assert "Komatsu" in params["q"]
    assert params["forms"] == "10-K"


def test_user_agent_header_contains_contact():
    client = EdgarClient()
    headers = client._auth_headers()
    assert "User-Agent" in headers
    assert "KomatsuIntel" in headers["User-Agent"]


@pytest.mark.asyncio
async def test_raises_client_error_on_4xx():
    client = EdgarClient()
    with patch.object(client, "get", new=AsyncMock(side_effect=ClientError(429, "Too Many Requests"))):
        with pytest.raises(ClientError) as exc_info:
            await client.search_filings("test")
    assert exc_info.value.status == 429
