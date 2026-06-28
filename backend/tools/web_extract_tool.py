import dataclasses
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator
from tools.base import BaseTool
from services.web_search_service import WebSearchService


class WebExtractInput(BaseModel):
    urls: list[str] = Field(
        min_length=1,
        max_length=20,
        description=(
            "URLs to extract content from (up to 20). "
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

    @field_validator("urls", mode="before")
    @classmethod
    def normalize_urls(cls, value):
        if isinstance(value, str):
            value = [u.strip() for u in value.split(",") if u.strip()]
        return value

    @field_validator("urls")
    @classmethod
    def validate_urls(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        for url in value:
            candidate = url.strip()
            parsed = urlparse(candidate)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")
            cleaned.append(candidate)
        return cleaned


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
        pages = await self._service.extract(
            urls=inp.urls,
            query=inp.query,
            extract_depth=inp.extract_depth,
        )
        return {"pages": [dataclasses.asdict(p) for p in pages], "count": len(pages)}
