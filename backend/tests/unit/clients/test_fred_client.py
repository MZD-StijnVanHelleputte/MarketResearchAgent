import pytest
from unittest.mock import AsyncMock, patch
from clients.fred_client import FredClient, get_fred_client


@pytest.mark.asyncio
async def test_get_observations_sends_correct_params():
    client = FredClient()
    raw = {"observations": [{"date": "2026-04-01", "value": "5.33"}]}

    with patch.object(client, "get", new=AsyncMock(return_value=raw)) as mock_get:
        result = await client.get_observations("FEDFUNDS", limit=5)

    mock_get.assert_called_once()
    params = mock_get.call_args[1]["params"]
    assert params["series_id"] == "FEDFUNDS"
    assert params["limit"] == 5
    assert params["file_type"] == "json"
    assert params["sort_order"] == "desc"
    assert result == raw


@pytest.mark.asyncio
async def test_get_observations_default_limit():
    client = FredClient()
    with patch.object(client, "get", new=AsyncMock(return_value={})) as mock_get:
        await client.get_observations("GDP")

    params = mock_get.call_args[1]["params"]
    assert params["limit"] == 10


@pytest.mark.asyncio
async def test_get_series_info_sends_correct_params():
    client = FredClient()
    raw = {"seriess": [{"title": "Federal Funds Rate", "units": "Percent"}]}

    with patch.object(client, "get", new=AsyncMock(return_value=raw)) as mock_get:
        result = await client.get_series_info("FEDFUNDS")

    params = mock_get.call_args[1]["params"]
    assert params["series_id"] == "FEDFUNDS"
    assert params["file_type"] == "json"
    assert result == raw


@pytest.mark.asyncio
async def test_auth_headers_returns_empty():
    client = FredClient()
    assert client._auth_headers() == {}


def test_get_fred_client_is_shared_singleton():
    """All FRED tools must share one client so the token-bucket rate limiter enforces
    a single global 120/min budget instead of one bucket per tool."""
    from services.fred_service import FredService
    from services.macro_service import MacroService

    shared = get_fred_client()
    assert get_fred_client() is shared
    assert FredService()._client is shared
    assert MacroService()._client is shared
