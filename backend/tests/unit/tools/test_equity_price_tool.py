import pytest
from unittest.mock import AsyncMock, MagicMock
from services.equity_service import EquityPrice
from tools.equity_price_tool import EquityPriceTool

PRICE = EquityPrice(ticker="CAT", price=350.0, currency="USD", market_cap_usd=175e9, date="2026-06-10")


def _mock_service(price=PRICE):
    svc = MagicMock()
    svc.get_price = AsyncMock(return_value=price)
    return svc


@pytest.mark.asyncio
async def test_run_returns_dict():
    tool = EquityPriceTool(service=_mock_service())
    result = await tool.run(ticker="CAT")
    assert result["ticker"] == "CAT"
    assert result["price"] == 350.0
    assert result["currency"] == "USD"


@pytest.mark.asyncio
async def test_run_delegates_to_service():
    svc = _mock_service()
    tool = EquityPriceTool(service=svc)
    await tool.run(ticker="CAT")
    svc.get_price.assert_called_once_with("CAT")


def test_tool_metadata():
    tool = EquityPriceTool()
    assert tool.name == "get_equity_price"
    assert "price" in tool.description.lower()
