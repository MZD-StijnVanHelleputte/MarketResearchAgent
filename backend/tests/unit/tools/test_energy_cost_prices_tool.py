from unittest.mock import AsyncMock, MagicMock

import pytest

from services.commodity_service import CommodityObservation, CommodityResult
from tools.energy_cost_prices_tool import EnergyCostPricesTool


RESULT = CommodityResult(
    symbol="WTI",
    endpoint="WTI",
    interval="monthly",
    unit="dollars per barrel",
    latest=CommodityObservation(date="2026-05-01", value=65.5),
    rows=[CommodityObservation(date="2026-05-01", value=65.5)],
)


def _mock_service() -> MagicMock:
    svc = MagicMock()
    svc.get_energy_cost_prices = AsyncMock(return_value=RESULT)
    return svc


@pytest.mark.asyncio
async def test_run_returns_normalized_dict():
    tool = EnergyCostPricesTool(service=_mock_service())
    result = await tool.run(symbol="WTI", interval="monthly")
    assert result["symbol"] == "WTI"
    assert result["latest"]["value"] == 65.5


@pytest.mark.asyncio
async def test_run_delegates_to_service():
    svc = _mock_service()
    tool = EnergyCostPricesTool(service=svc)
    await tool.run(symbol="BRENT", interval="weekly")
    svc.get_energy_cost_prices.assert_called_once_with("BRENT", "weekly")


def test_tool_metadata():
    tool = EnergyCostPricesTool()
    assert tool.name == "get_energy_cost_prices"
    assert "energy" in tool.description.lower()
