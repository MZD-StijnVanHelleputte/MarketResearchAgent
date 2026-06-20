from unittest.mock import AsyncMock, MagicMock

import pytest

from services.commodity_service import CommodityObservation, CommodityResult
from tools.mining_metals_prices_tool import MiningMetalsPricesTool


RESULT = CommodityResult(
    symbol="COPPER",
    endpoint="COPPER",
    interval="monthly",
    unit="dollars per metric ton",
    latest=CommodityObservation(date="2026-05-01", value=9500.0),
    rows=[CommodityObservation(date="2026-05-01", value=9500.0)],
)


def _mock_service() -> MagicMock:
    svc = MagicMock()
    svc.get_mining_metals_prices = AsyncMock(return_value=RESULT)
    return svc


@pytest.mark.asyncio
async def test_run_returns_normalized_dict():
    tool = MiningMetalsPricesTool(service=_mock_service())
    result = await tool.run(symbol="COPPER", interval="monthly")
    assert result["symbol"] == "COPPER"
    assert result["endpoint"] == "COPPER"
    assert result["latest"]["value"] == 9500.0
    assert result["source"] == "alpha_vantage"


@pytest.mark.asyncio
async def test_run_delegates_to_service():
    svc = _mock_service()
    tool = MiningMetalsPricesTool(service=svc)
    await tool.run(symbol="GOLD", interval="daily", include_history=False)
    svc.get_mining_metals_prices.assert_called_once_with("GOLD", "daily", False)


def test_tool_metadata():
    tool = MiningMetalsPricesTool()
    assert tool.name == "get_mining_metals_prices"
    assert "alpha vantage" in tool.description.lower()
