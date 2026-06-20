import pytest
from unittest.mock import AsyncMock, MagicMock
from services.competition_service import CompanyFinancials
from tools.company_financials_tool import CompanyFinancialsTool

FINANCIALS = CompanyFinancials(
    ticker="CAT", name="Caterpillar Inc",
    revenue_usd=67e9, net_income_usd=6e9, capex_usd=2e9,
    market_cap_usd=150e9, pe_ratio=15.2, date="2024-12-31",
)


def _mock_service(result=FINANCIALS):
    svc = MagicMock()
    svc.get_financials = AsyncMock(return_value=result)
    return svc


@pytest.mark.asyncio
async def test_run_returns_dict():
    tool = CompanyFinancialsTool(service=_mock_service())
    result = await tool.run(ticker="CAT")
    assert result["ticker"] == "CAT"
    assert result["name"] == "Caterpillar Inc"
    assert result["revenue_usd"] == 67e9


@pytest.mark.asyncio
async def test_run_delegates_to_service():
    svc = _mock_service()
    tool = CompanyFinancialsTool(service=svc)
    await tool.run(ticker="CAT")
    svc.get_financials.assert_called_once_with("CAT")


def test_tool_metadata():
    tool = CompanyFinancialsTool()
    assert tool.name == "get_company_financials"
    assert "financial" in tool.description.lower()
