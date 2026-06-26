import pytest
from unittest.mock import AsyncMock, MagicMock
from services.web_search_service import SearchResult
from tools.web_search_tool import WebSearchTool

RESULTS = [
    SearchResult(title="Komatsu Mining", url="https://example.com", snippet="Leader in mining equipment", score=0.9),
]


def _mock_service(results=RESULTS):
    svc = MagicMock()
    svc.search = AsyncMock(return_value=results)
    return svc


@pytest.mark.asyncio
async def test_run_returns_results():
    tool = WebSearchTool(service=_mock_service())
    result = await tool.run(query="Komatsu mining")
    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["title"] == "Komatsu Mining"


@pytest.mark.asyncio
async def test_run_delegates_to_service():
    svc = _mock_service()
    tool = WebSearchTool(service=svc)
    await tool.run(query="copper price", max_results=3)
    # The tool forwards the full Tavily param set; assert the key args are present.
    svc.search.assert_called_once()
    kwargs = svc.search.call_args.kwargs
    assert kwargs["query"] == "copper price"
    assert kwargs["max_results"] == 3


def test_tool_metadata():
    tool = WebSearchTool()
    assert tool.name == "web_search"
    assert "web" in tool.description.lower() or "search" in tool.description.lower()
