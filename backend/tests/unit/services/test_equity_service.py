import pytest
from unittest.mock import AsyncMock, MagicMock
from services.equity_service import (
    EquityService, EquityPrice, EquityHistory, EquityFinancials, ServiceError,
)

PRICE_RAW = {"symbol": "CAT", "price": 350.0, "currency": "USD", "market_cap": 175e9, "date": "2026-06-10"}
HISTORY_RAW = [
    {"date": "2026-01-02", "open": 340.0, "high": 355.0, "low": 338.0, "close": 350.0, "volume": 2000000}
]
FINANCIALS_RAW = [
    {"date": "2025-12-31", "Total Revenue": 1000.0, "Net Income": 100.0},
]


def _mock_client(price=PRICE_RAW, history=HISTORY_RAW, financials=FINANCIALS_RAW):
    client = MagicMock()
    client.get_price = AsyncMock(return_value=price)
    client.get_history = AsyncMock(return_value=history)
    client.get_financials = AsyncMock(return_value=financials)
    return client


@pytest.mark.asyncio
async def test_get_price_returns_equity_price():
    svc = EquityService(client=_mock_client())
    result = await svc.get_price("CAT")
    assert isinstance(result, EquityPrice)
    assert result.ticker == "CAT"
    assert result.price == 350.0
    assert result.currency == "USD"
    assert result.market_cap_usd == 175e9


@pytest.mark.asyncio
async def test_get_history_returns_equity_history():
    svc = EquityService(client=_mock_client())
    result = await svc.get_history("CAT", period="1mo")
    assert isinstance(result, EquityHistory)
    assert result.ticker == "CAT"
    assert result.period == "1mo"
    assert len(result.rows) == 1


@pytest.mark.asyncio
async def test_get_financials_returns_equity_financials():
    svc = EquityService(client=_mock_client())
    result = await svc.get_financials("CAT", period="annual")
    assert isinstance(result, EquityFinancials)
    assert result.ticker == "CAT"
    assert result.period == "annual"
    assert len(result.rows) == 1


@pytest.mark.asyncio
async def test_get_financials_raises_service_error_on_exception():
    client = MagicMock()
    client.get_financials = AsyncMock(side_effect=RuntimeError("yfinance unavailable"))
    svc = EquityService(client=client)
    with pytest.raises(ServiceError):
        await svc.get_financials("CAT")


@pytest.mark.asyncio
async def test_get_price_raises_service_error_on_exception():
    client = MagicMock()
    client.get_price = AsyncMock(side_effect=RuntimeError("yfinance unavailable"))
    svc = EquityService(client=client)
    with pytest.raises(ServiceError):
        await svc.get_price("CAT")
