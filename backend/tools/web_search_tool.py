import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.web_search_service import WebSearchService


class WebSearchInput(BaseModel):
    query: str = Field(description="Search keywords or question (under 400 characters).")
    max_results: int = Field(default=5, ge=1, le=20, description="Number of results (1–20).")
    search_depth: str = Field(
        default="basic",
        pattern=r"^(ultra-fast|fast|basic|advanced)$",
        description=(
            "Search precision: 'basic' (default, fast), 'advanced' (slower, higher precision for "
            "specific facts), 'fast' (good with chunks), 'ultra-fast' (real-time)."
        ),
    )
    topic: str = Field(
        default="general",
        pattern=r"^(general|news|finance)$",
        description="Topic filter: 'general' (default), 'news' (recent events), 'finance' (financial data).",
    )
    time_range: str | None = Field(
        default=None,
        description="Restrict results to: 'day', 'week', 'month', or 'year'. Optional.",
    )
    include_domains: str = Field(
        default="",
        description=(
            "Comma-separated domain allowlist, e.g. 'sec.gov,reuters.com'. "
            "Use to focus on trusted sources. Optional."
        ),
    )
    exclude_domains: str = Field(
        default="",
        description="Comma-separated domain blocklist. Optional.",
    )
    include_answer: bool = Field(
        default=False,
        description="When true, Tavily prepends an AI-generated answer summary to the results.",
    )
    include_raw_content: bool = Field(
        default=False,
        description="When true, returns full page text alongside snippets (increases response size).",
    )


class WebSearchTool(BaseTool):
    name = "web_search"
    description = (
        "Search the open web via Tavily. Returns titles, URLs, and content snippets ranked by "
        "relevance. Use search_depth='advanced' for precision lookups, topic='news' for recent "
        "events, topic='finance' for financial data, time_range to restrict recency, and "
        "include_domains to focus on trusted sources (e.g. 'sec.gov,reuters.com')."
    )
    input_schema = WebSearchInput

    def __init__(self, service: WebSearchService | None = None) -> None:
        self._service = service or WebSearchService()

    async def run(self, **kwargs) -> dict:
        inp = WebSearchInput(**kwargs)
        include_domains = [d.strip() for d in inp.include_domains.split(",") if d.strip()] or None
        exclude_domains = [d.strip() for d in inp.exclude_domains.split(",") if d.strip()] or None
        results = await self._service.search(
            query=inp.query,
            max_results=inp.max_results,
            search_depth=inp.search_depth,
            topic=inp.topic,
            time_range=inp.time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            include_answer=inp.include_answer,
            include_raw_content=inp.include_raw_content,
        )
        return {"results": [dataclasses.asdict(r) for r in results]}
