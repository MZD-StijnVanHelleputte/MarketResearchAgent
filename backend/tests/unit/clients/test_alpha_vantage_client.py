from unittest.mock import AsyncMock, patch

import pytest

from clients.alpha_vantage_client import AlphaVantageClient
from clients.base_http_client import ClientError


RAW = {"data": [{"date": "2026-05-01", "value": "9500"}]}


@pytest.mark.asyncio
async def test_get_global_quote_uses_global_quote_function():
    client = AlphaVantageClient()
    with patch.object(client, "get", new=AsyncMock(return_value=RAW)) as mock_get:
        result = await client.get_global_quote("FCX")

    mock_get.assert_called_once()
    call_path, call_kwargs = mock_get.call_args[0][0], mock_get.call_args[1]
    assert call_path == "/query"
    params = call_kwargs["params"]
    assert params["function"] == "GLOBAL_QUOTE"
    assert params["symbol"] == "FCX"
    assert "apikey" in params
    assert result == RAW


@pytest.mark.asyncio
async def test_get_gold_silver_spot_uses_spot_function():
    client = AlphaVantageClient()
    with patch.object(client, "get", new=AsyncMock(return_value=RAW)) as mock_get:
        result = await client.get_gold_silver_spot("GOLD")

    params = mock_get.call_args.kwargs["params"]
    assert params["function"] == "GOLD_SILVER_SPOT"
    assert params["symbol"] == "GOLD"
    assert result == RAW


@pytest.mark.asyncio
async def test_get_gold_silver_history_uses_history_function():
    client = AlphaVantageClient()
    with patch.object(client, "get", new=AsyncMock(return_value=RAW)) as mock_get:
        result = await client.get_gold_silver_history("SILVER", "daily")

    params = mock_get.call_args.kwargs["params"]
    assert params["function"] == "GOLD_SILVER_HISTORY"
    assert params["symbol"] == "SILVER"
    assert params["interval"] == "daily"
    assert result == RAW


@pytest.mark.asyncio
async def test_get_commodity_series_uses_function_and_interval():
    client = AlphaVantageClient()
    with patch.object(client, "get", new=AsyncMock(return_value=RAW)) as mock_get:
        result = await client.get_commodity_series("COPPER", "quarterly")

    params = mock_get.call_args.kwargs["params"]
    assert params["function"] == "COPPER"
    assert params["interval"] == "quarterly"
    assert result == RAW


@pytest.mark.asyncio
async def test_get_commodity_series_raises_client_error_on_4xx():
    client = AlphaVantageClient()
    with patch.object(client, "get", new=AsyncMock(side_effect=ClientError(403, "Forbidden"))):
        with pytest.raises(ClientError) as exc_info:
            await client.get_commodity_series("COPPER")
    assert exc_info.value.status == 403
