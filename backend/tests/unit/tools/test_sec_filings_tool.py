import pytest
from unittest.mock import AsyncMock, MagicMock
from services.competition_service import Filing
from tools.sec_filings_tool import SecFilingsTool

FILINGS = [
    Filing(entity_name="Caterpillar Inc", form_type="10-K", file_date="2024-02-14", period="2023-12-31", snippet=""),
]


def _mock_service(filings=FILINGS):
    svc = MagicMock()
    svc.get_filings = AsyncMock(return_value=filings)
    return svc


@pytest.mark.asyncio
async def test_run_returns_filings_dict():
    tool = SecFilingsTool(service=_mock_service())
    result = await tool.run(query="Caterpillar mining capex")
    assert "filings" in result
    assert len(result["filings"]) == 1
    assert result["filings"][0]["entity_name"] == "Caterpillar Inc"


@pytest.mark.asyncio
async def test_run_delegates_to_service():
    svc = _mock_service()
    tool = SecFilingsTool(service=svc)
    await tool.run(query="test", forms="10-K")
    svc.get_filings.assert_called_once_with(query="test", forms="10-K")


def test_tool_metadata():
    tool = SecFilingsTool()
    assert tool.name == "search_sec_filings"
    assert "sec" in tool.description.lower() or "edgar" in tool.description.lower()
