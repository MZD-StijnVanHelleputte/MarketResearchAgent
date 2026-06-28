import pytest
from pydantic import ValidationError
from unittest.mock import AsyncMock, MagicMock

from services.web_search_service import ExtractedPage
from tools.web_extract_tool import WebExtractInput, WebExtractTool


PAGES = [
    ExtractedPage(
        url="https://investor.komatsu.com",
        content="Komatsu investor relations",
        images=[],
    )
]


def _mock_service(pages=PAGES):
    svc = MagicMock()
    svc.extract = AsyncMock(return_value=pages)
    return svc


@pytest.mark.asyncio
async def test_run_accepts_url_list_and_delegates_to_service():
    svc = _mock_service()
    tool = WebExtractTool(service=svc)

    result = await tool.run(
        urls=["https://investor.komatsu.com", "https://investor.caterpillar.com"],
        query="investor relations",
        extract_depth="advanced",
    )

    svc.extract.assert_called_once_with(
        urls=["https://investor.komatsu.com", "https://investor.caterpillar.com"],
        query="investor relations",
        extract_depth="advanced",
    )
    assert result["count"] == 1
    assert result["pages"][0]["url"] == "https://investor.komatsu.com"


@pytest.mark.asyncio
async def test_run_accepts_legacy_comma_separated_urls():
    svc = _mock_service()
    tool = WebExtractTool(service=svc)

    await tool.run(
        urls="https://investor.komatsu.com, https://investor.caterpillar.com",
    )

    kwargs = svc.extract.call_args.kwargs
    assert kwargs["urls"] == [
        "https://investor.komatsu.com",
        "https://investor.caterpillar.com",
    ]


@pytest.mark.parametrize(
    "urls",
    [
        [],
        "",
        ["not-a-url"],
        ["ftp://example.com/file"],
    ],
)
def test_input_rejects_empty_or_invalid_urls(urls):
    with pytest.raises(ValidationError):
        WebExtractInput(urls=urls)

