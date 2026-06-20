from unittest.mock import AsyncMock, MagicMock

import pytest

from clients.base_http_client import ClientError
from services.commodity_service import CommodityResult, CommodityService, ServiceError


SERIES_RAW = {
    "name": "Global price of Copper",
    "interval": "monthly",
    "unit": "U.S. dollars per metric ton",
    "data": [
        {"date": "2026-05-01", "value": "9500.5"},
        {"date": "2026-04-01", "value": "."},
    ],
}

SPOT_RAW = {
    "symbol": "GOLD",
    "price": "2350.25",
    "timestamp": "2026-06-10 12:00:00",
    "unit": "USD per troy ounce",
}

HISTORY_RAW = {
    "Meta Data": {"Unit": "USD per troy ounce"},
    "data": [
        {"date": "2026-06-10", "close": "31.50"},
    ],
}


def _mock_client() -> MagicMock:
    client = MagicMock()
    client.get_commodity_series = AsyncMock(return_value=SERIES_RAW)
    client.get_gold_silver_spot = AsyncMock(return_value=SPOT_RAW)
    client.get_gold_silver_history = AsyncMock(return_value=HISTORY_RAW)
    return client


@pytest.mark.asyncio
async def test_get_mining_metals_prices_parses_series_latest():
    client = _mock_client()
    svc = CommodityService(client=client)
    result = await svc.get_mining_metals_prices("COPPER", "monthly")
    assert isinstance(result, CommodityResult)
    assert result.symbol == "COPPER"
    assert result.endpoint == "COPPER"
    assert result.unit == "U.S. dollars per metric ton"
    assert result.latest.value == 9500.5
    assert result.rows[1].value is None
    client.get_commodity_series.assert_called_once_with("COPPER", "monthly")


@pytest.mark.asyncio
async def test_get_mining_metals_prices_uses_gold_silver_history_by_default():
    client = _mock_client()
    svc = CommodityService(client=client)
    result = await svc.get_mining_metals_prices("SILVER", "daily")
    assert result.endpoint == "GOLD_SILVER_HISTORY"
    assert result.latest.value == 31.5
    client.get_gold_silver_history.assert_called_once_with("SILVER", "daily")


@pytest.mark.asyncio
async def test_get_mining_metals_prices_uses_gold_silver_spot_when_requested():
    client = _mock_client()
    svc = CommodityService(client=client)
    result = await svc.get_mining_metals_prices("GOLD", "monthly", include_history=False)
    assert result.endpoint == "GOLD_SILVER_SPOT"
    assert result.interval == "spot"
    assert result.latest.value == 2350.25
    client.get_gold_silver_spot.assert_called_once_with("GOLD")


@pytest.mark.asyncio
async def test_get_energy_cost_prices_parses_series():
    client = _mock_client()
    svc = CommodityService(client=client)
    result = await svc.get_energy_cost_prices("WTI", "weekly")
    assert result.symbol == "WTI"
    assert result.latest.value == 9500.5
    client.get_commodity_series.assert_called_once_with("WTI", "weekly")


@pytest.mark.asyncio
async def test_get_broad_commodity_cycle_uses_all_commodities():
    client = _mock_client()
    svc = CommodityService(client=client)
    result = await svc.get_broad_commodity_cycle("annual")
    assert result.symbol == "ALL_COMMODITIES"
    client.get_commodity_series.assert_called_once_with("ALL_COMMODITIES", "annual")


@pytest.mark.asyncio
async def test_notice_body_raises_service_error():
    client = _mock_client()
    client.get_commodity_series = AsyncMock(return_value={"Note": "rate limit"})
    svc = CommodityService(client=client)
    with pytest.raises(ServiceError, match="rate limit"):
        await svc.get_mining_metals_prices("COPPER")


@pytest.mark.asyncio
async def test_client_error_raises_service_error():
    client = _mock_client()
    client.get_commodity_series = AsyncMock(side_effect=ClientError(403, "Forbidden"))
    svc = CommodityService(client=client)
    with pytest.raises(ServiceError, match="Alpha Vantage request failed"):
        await svc.get_mining_metals_prices("COPPER")


@pytest.mark.asyncio
async def test_invalid_symbol_raises_service_error():
    svc = CommodityService(client=_mock_client())
    with pytest.raises(ServiceError, match="Unsupported mining metals symbol"):
        await svc.get_mining_metals_prices("IRON_ORE")


@pytest.mark.asyncio
async def test_invalid_interval_raises_service_error():
    svc = CommodityService(client=_mock_client())
    with pytest.raises(ServiceError, match="Unsupported interval"):
        await svc.get_energy_cost_prices("WTI", "annual")
