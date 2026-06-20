import pytest
from unittest.mock import AsyncMock, patch
from clients.fmp_client import FmpClient
from clients.base_http_client import ClientError

PROFILE_RESPONSE = [{"companyName": "Caterpillar Inc", "mktCap": 150_000_000_000, "pe": 15.2}]
METRICS_RESPONSE = [{"revenueTTM": 67_000_000_000, "netIncomeTTM": 6_000_000_000}]


@pytest.mark.asyncio
async def test_get_company_profile_returns_dict():
    client = FmpClient()
    with patch.object(client, "get", new=AsyncMock(return_value=PROFILE_RESPONSE)) as mock_get:
        result = await client.get_company_profile("CAT")

    mock_get.assert_called_once()
    path = mock_get.call_args[0][0]
    assert path == "/v3/profile/CAT"
    params = mock_get.call_args[1]["params"]
    assert "apikey" in params
    assert result == PROFILE_RESPONSE


@pytest.mark.asyncio
async def test_get_key_metrics_returns_dict():
    client = FmpClient()
    with patch.object(client, "get", new=AsyncMock(return_value=METRICS_RESPONSE)) as mock_get:
        result = await client.get_key_metrics("CAT")

    assert "/v3/key-metrics-ttm/CAT" in mock_get.call_args[0][0]
    assert result == METRICS_RESPONSE


@pytest.mark.asyncio
async def test_get_income_statement_passes_limit():
    client = FmpClient()
    with patch.object(client, "get", new=AsyncMock(return_value=[])) as mock_get:
        await client.get_income_statement("CAT", limit=2)

    params = mock_get.call_args[1]["params"]
    assert params["limit"] == 2


@pytest.mark.asyncio
async def test_raises_client_error_on_4xx():
    client = FmpClient()
    with patch.object(client, "get", new=AsyncMock(side_effect=ClientError(401, "Unauthorized"))):
        with pytest.raises(ClientError) as exc_info:
            await client.get_company_profile("CAT")
    assert exc_info.value.status == 401
