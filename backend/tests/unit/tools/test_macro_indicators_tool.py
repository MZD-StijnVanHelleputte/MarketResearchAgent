import pytest
from unittest.mock import AsyncMock, MagicMock
from services.macro_service import MacroIndicator, MacroObservation
from tools.macro_indicators_tool import MacroIndicatorsTool


def _mock_service(indicator: MacroIndicator | None = None):
    svc = MagicMock()
    svc.get_indicator = AsyncMock(
        return_value=indicator or MacroIndicator(
            series_id="FEDFUNDS",
            title="Federal Funds Effective Rate",
            units="Percent",
            frequency="M",
            observations=[MacroObservation(date="2026-04-01", value=5.33)],
        )
    )
    return svc


@pytest.mark.asyncio
async def test_run_returns_dict():
    tool = MacroIndicatorsTool(service=_mock_service())
    result = await tool.run(series_id="FEDFUNDS")

    assert isinstance(result, dict)
    assert result["series_id"] == "FEDFUNDS"
    assert result["title"] == "Federal Funds Effective Rate"
    assert len(result["observations"]) == 1
    assert result["observations"][0]["value"] == 5.33


@pytest.mark.asyncio
async def test_run_passes_series_id_and_limit():
    svc = _mock_service()
    tool = MacroIndicatorsTool(service=svc)
    await tool.run(series_id="GDP", limit=5)

    svc.get_indicator.assert_called_once_with("GDP", 5)


@pytest.mark.asyncio
async def test_run_uses_default_limit():
    svc = _mock_service()
    tool = MacroIndicatorsTool(service=svc)
    await tool.run(series_id="CPIAUCSL")

    svc.get_indicator.assert_called_once_with("CPIAUCSL", 10)
