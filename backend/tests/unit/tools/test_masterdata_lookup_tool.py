import pytest
from unittest.mock import MagicMock
from tools.masterdata_lookup_tool import MasterdataLookupTool


@pytest.fixture
def mock_service():
    svc = MagicMock()
    svc.lookup.return_value = [
        {"name": "Hastings Deering", "region": "Asia-Pacific", "countries": ["Australia"], "products": ["mining"]},
        {"name": "WesTrac", "region": "Asia-Pacific", "countries": ["Australia", "China"], "products": ["mining"]},
    ]
    return svc


@pytest.fixture
def tool(mock_service):
    return MasterdataLookupTool(service=mock_service)


@pytest.mark.asyncio
async def test_run_returns_results(tool, mock_service):
    result = await tool.run(entity_type="distributors")
    assert result["entity_type"] == "distributors"
    assert len(result["results"]) == 2
    mock_service.lookup.assert_called_once_with("distributors", "", "")


@pytest.mark.asyncio
async def test_run_passes_region_filter(tool, mock_service):
    await tool.run(entity_type="distributors", region="Asia-Pacific")
    mock_service.lookup.assert_called_once_with("distributors", "Asia-Pacific", "")


@pytest.mark.asyncio
async def test_run_passes_keyword_filter(tool, mock_service):
    await tool.run(entity_type="competitors", keyword="CAT")
    mock_service.lookup.assert_called_once_with("competitors", "", "CAT")


@pytest.mark.asyncio
async def test_run_passes_construction_entity_type(tool, mock_service):
    await tool.run(entity_type="construction")
    mock_service.lookup.assert_called_once_with("construction", "", "")


@pytest.mark.asyncio
async def test_run_passes_others_entity_type(tool, mock_service):
    await tool.run(entity_type="others")
    mock_service.lookup.assert_called_once_with("others", "", "")


@pytest.mark.asyncio
async def test_run_unknown_entity_type_propagates_error(mock_service):
    mock_service.lookup.side_effect = ValueError("Unknown entity_type 'invalid'")
    tool = MasterdataLookupTool(service=mock_service)
    with pytest.raises(ValueError, match="Unknown entity_type"):
        await tool.run(entity_type="invalid")
