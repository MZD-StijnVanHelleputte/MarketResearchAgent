import pytest
from unittest.mock import AsyncMock, MagicMock
from services.news_service import NewsSource
from tools.news_sources_tool import NewsSourcesTool


def _mock_service(sources: list[NewsSource]):
    svc = MagicMock()
    svc.list_sources = AsyncMock(return_value=sources)
    return svc


SOURCES = [
    NewsSource(
        id="bbc-news",
        name="BBC News",
        description="Global news coverage.",
        url="https://bbc.co.uk",
        category="general",
        language="en",
        country="gb",
    )
]


@pytest.mark.asyncio
async def test_run_returns_sources_dict():
    tool = NewsSourcesTool(service=_mock_service(SOURCES))
    result = await tool.run(category="general")

    assert "sources" in result
    assert len(result["sources"]) == 1
    assert result["sources"][0]["id"] == "bbc-news"
    assert result["sources"][0]["name"] == "BBC News"


@pytest.mark.asyncio
async def test_run_passes_all_args_to_service():
    svc = _mock_service([])
    tool = NewsSourcesTool(service=svc)
    await tool.run(category="technology", language="en", country="gb")

    svc.list_sources.assert_called_once_with(
        category="technology",
        language="en",
        country="gb",
    )


@pytest.mark.asyncio
async def test_run_empty_results():
    tool = NewsSourcesTool(service=_mock_service([]))
    result = await tool.run()
    assert result == {"sources": []}


def test_tool_metadata():
    tool = NewsSourcesTool()
    assert tool.name == "news_sources"
    assert "source" in tool.description.lower()
