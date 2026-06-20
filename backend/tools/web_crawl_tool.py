import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.web_search_service import WebSearchService


class WebCrawlInput(BaseModel):
    url: str = Field(
        description=(
            "Root URL to start crawling from, e.g. 'https://www.cat.com/en_US/news/' "
            "or 'https://www.mining.com/projects/'. The crawler follows links within "
            "the same domain up to max_depth levels deep."
        )
    )
    instructions: str = Field(
        default="",
        description=(
            "Natural language semantic focus for the crawl. When provided, Tavily returns "
            "only relevant content chunks instead of full pages — prevents context explosion. "
            "Example: 'Find product announcements and mining equipment launches'."
        ),
    )
    max_depth: int = Field(
        default=1,
        ge=1,
        le=5,
        description="How many link-levels deep to crawl (1–5). Start at 1 and increase if needed.",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum number of pages to crawl (1–50). Keep low to avoid runaway crawls.",
    )
    select_paths: str = Field(
        default="",
        description=(
            "Comma-separated regex patterns to restrict crawled paths, "
            "e.g. '/news/.*,/press-releases/.*'. Optional."
        ),
    )


class WebCrawlTool(BaseTool):
    name = "web_crawl"
    description = (
        "Systematically crawl a website and extract content from multiple pages via Tavily POST /crawl. "
        "Use to bulk-extract a competitor's IR/news section, a mine operator's project pages, "
        "or a regulatory body's recent publications. "
        "Set instructions to a semantic focus to get only relevant chunks (avoids context overflow). "
        "Use web_map first to discover URL structure before committing to a crawl. "
        "Returns pages_crawled count and a list of pages with url and content."
    )
    input_schema = WebCrawlInput

    def __init__(self, service: WebSearchService | None = None) -> None:
        self._service = service or WebSearchService()

    async def run(self, **kwargs) -> dict:
        inp = WebCrawlInput(**kwargs)
        select = [p.strip() for p in inp.select_paths.split(",") if p.strip()] or None
        result = await self._service.crawl(
            url=inp.url,
            instructions=inp.instructions,
            max_depth=inp.max_depth,
            limit=inp.limit,
            select_paths=select,
        )
        return dataclasses.asdict(result)
