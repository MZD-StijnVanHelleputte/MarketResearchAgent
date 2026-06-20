import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.web_search_service import WebSearchService


class WebExtractInput(BaseModel):
    urls: str = Field(
        description=(
            "Comma-separated URLs to extract content from (up to 20). "
            "Typically used after web_search returns links you want to read in full — "
            "e.g. competitor press releases, mining project announcement pages, "
            "regulatory filings or IR pages."
        )
    )
    query: str = Field(
        default="",
        description=(
            "Optional relevance query. When provided, Tavily reranks the extracted chunks "
            "to surface the most query-relevant content. Leave empty for full page content."
        ),
    )
    extract_depth: str = Field(
        default="basic",
        pattern=r"^(basic|advanced)$",
        description=(
            "'basic' (default) — fast, works for most pages. "
            "'advanced' — for JavaScript-rendered SPAs and dynamic content."
        ),
    )


class WebExtractTool(BaseTool):
    name = "web_extract"
    description = (
        "Extract clean full-page content from specific URLs via Tavily POST /extract. "
        "Use this after web_search returns URLs you want to read in full: competitor press "
        "releases, mining project pages, IR sections, regulatory publications. "
        "Handles JavaScript-rendered pages with extract_depth='advanced'. "
        "Returns a list of extracted pages with url, content, and images."
    )
    input_schema = WebExtractInput

    def __init__(self, service: WebSearchService | None = None) -> None:
        self._service = service or WebSearchService()

    async def run(self, **kwargs) -> dict:
        inp = WebExtractInput(**kwargs)
        url_list = [u.strip() for u in inp.urls.split(",") if u.strip()]
        pages = await self._service.extract(
            urls=url_list,
            query=inp.query,
            extract_depth=inp.extract_depth,
        )
        return {"pages": [dataclasses.asdict(p) for p in pages], "count": len(pages)}
