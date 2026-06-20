from unittest.mock import AsyncMock, MagicMock

import pytest

from services.commodity_service import CommodityObservation, CommodityResult
from tools.broad_commodity_cycle_tool import BroadCommodityCycleTool


RESULT = CommodityResult(
    symbol="ALL_COMMODITIES",
    endpoint="ALL_COMMODITIES",
    interval="monthly",
    unit="index",
    latest=CommodityObservation(date="2026-05-01", value=180.2),
    rows=[CommodityObservation(date="2026-05-01", value=180.2)],
)


def _mock_service() -> MagicMock:
    svc = MagicMock()
    svc.get_broad_commodity_cycle = AsyncMock(return_value=RESULT)
    return svc


@pytest.mark.asyncio
async def test_run_returns_normalized_dict():
    tool = BroadCommodityCycleTool(service=_mock_service())
    result = await tool.run(interval="monthly")
    assert result["symbol"] == "ALL_COMMODITIES"
    assert result["latest"]["value"] == 180.2


@pytest.mark.asyncio
async def test_run_delegates_to_service():
    svc = _mock_service()
    tool = BroadCommodityCycleTool(service=svc)
    await tool.run(interval="annual")
    svc.get_broad_commodity_cycle.assert_called_once_with("annual")


def test_tool_metadata():
    tool = BroadCommodityCycleTool()
    assert tool.name == "get_broad_commodity_cycle"
    assert "all_commodities" in tool.description.lower()
