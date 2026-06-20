import pytest
from unittest.mock import AsyncMock, MagicMock
from services.equity_service import EquityFinancials
from tools.equity_financials_tool import EquityFinancialsTool

FINANCIALS = EquityFinancials(
    ticker="CAT",
    period="annual",
    rows=[{"date": "2025-12-31", "Total Revenue": 1000.0, "Net Income": 100.0}],
)


def _mock_service(financials=FINANCIALS):
    svc = MagicMock()
    svc.get_financials = AsyncMock(return_value=financials)
    return svc


@pytest.mark.asyncio
async def test_run_returns_dict():
    tool = EquityFinancialsTool(service=_mock_service())
    result = await tool.run(ticker="CAT")
    assert result["ticker"] == "CAT"
    assert result["period"] == "annual"
    assert result["rows"][0]["Total Revenue"] == 1000.0


@pytest.mark.asyncio
async def test_run_delegates_to_service_with_period():
    svc = _mock_service()
    tool = EquityFinancialsTool(service=svc)
    await tool.run(ticker="CAT", period="quarterly")
    svc.get_financials.assert_called_once_with("CAT", "quarterly")


@pytest.mark.asyncio
async def test_run_defaults_to_annual():
    svc = _mock_service()
    tool = EquityFinancialsTool(service=svc)
    await tool.run(ticker="CAT")
    svc.get_financials.assert_called_once_with("CAT", "annual")


def test_tool_metadata():
    tool = EquityFinancialsTool()
    assert tool.name == "get_equity_financials"
    assert "financial" in tool.description.lower()
