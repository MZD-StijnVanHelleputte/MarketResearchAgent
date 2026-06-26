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


TICKER_MAP_RESPONSE = {
    "0": {"cik_str": 831259, "ticker": "FCX", "title": "FREEPORT-MCMORAN INC"},
    "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
}


@pytest.mark.asyncio
async def test_get_cik_for_ticker_resolves_and_pads():
    client = EdgarClient()
    with patch.object(client, "get", new=AsyncMock(return_value=TICKER_MAP_RESPONSE)):
        cik = await client.get_cik_for_ticker("fcx")
    assert cik == "0000831259"


@pytest.mark.asyncio
async def test_get_cik_for_ticker_unknown_returns_none():
    client = EdgarClient()
    with patch.object(client, "get", new=AsyncMock(return_value=TICKER_MAP_RESPONSE)):
        cik = await client.get_cik_for_ticker("NOPE")
    assert cik is None


@pytest.mark.asyncio
async def test_get_cik_for_ticker_caches_after_first_fetch():
    client = EdgarClient()
    mock_get = AsyncMock(return_value=TICKER_MAP_RESPONSE)
    with patch.object(client, "get", new=mock_get):
        await client.get_cik_for_ticker("FCX")
        await client.get_cik_for_ticker("NVDA")
    mock_get.assert_called_once()


@pytest.mark.asyncio
async def test_get_submissions_calls_correct_url():
    client = EdgarClient()
    with patch.object(client, "get", new=AsyncMock(return_value={"name": "FCX"})) as mock_get:
        result = await client.get_submissions("0000831259")

    mock_get.assert_called_once_with("https://data.sec.gov/submissions/CIK0000831259.json")
    assert result == {"name": "FCX"}


@pytest.mark.asyncio
async def test_get_document_bytes_uses_get_bytes():
    client = EdgarClient()
    with patch.object(client, "get_bytes", new=AsyncMock(return_value=b"raw")) as mock_get_bytes:
        result = await client.get_document_bytes("https://www.sec.gov/Archives/edgar/data/1/a.htm")

    mock_get_bytes.assert_called_once_with("https://www.sec.gov/Archives/edgar/data/1/a.htm")
    assert result == b"raw"
