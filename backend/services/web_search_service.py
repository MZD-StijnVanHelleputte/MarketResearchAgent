import logging
from dataclasses import dataclass, field

from clients.web_search_client import WebResearchClient, WebSearchClient
from clients.base_http_client import ClientError

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    pass


# --- Result dataclasses ---

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float
    raw_content: str = ""


@dataclass
class ExtractedPage:
    url: str
    content: str
    images: list[str] = field(default_factory=list)


@dataclass
class CrawlPage:
    url: str
    content: str


@dataclass
class CrawlResult:
    root_url: str
    pages_crawled: int
    pages: list[CrawlPage] = field(default_factory=list)


@dataclass
class MapResult:
    root_url: str
    urls: list[str] = field(default_factory=list)


@dataclass
class ResearchReport:
    query: str
    model: str
    report: str
    citations: list[str] = field(default_factory=list)


# --- Service ---

class WebSearchService:
    """Business logic wrapper over Tavily API (WebSearchClient + WebResearchClient)."""

    def __init__(
        self,
        client: WebSearchClient | None = None,
        research_client: WebResearchClient | None = None,
    ) -> None:
        self._client = client or WebSearchClient()
        self._research_client = research_client or WebResearchClient()

    async def search(
        self,
        query: str,
        max_results: int = 5,
        search_depth: str = "basic",
        topic: str = "general",
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        include_answer: bool = False,
        include_raw_content: bool = False,
    ) -> list[SearchResult]:
        try:
            raw = await self._client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                topic=topic,
                time_range=time_range,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
                include_answer=include_answer,
                include_raw_content=include_raw_content,
            )
        except ClientError as exc:
            raise ServiceError(f"Web search failed: {exc}") from exc

        results: list[SearchResult] = []
        for item in raw.get("results", []):
            results.append(SearchResult(
                title=item.get("title") or "",
                url=item.get("url") or "",
                snippet=item.get("content") or item.get("snippet") or "",
                score=float(item.get("score", 0.0)),
                raw_content=item.get("raw_content") or "",
            ))
        return results

    async def extract(
        self,
        urls: list[str],
        query: str = "",
        extract_depth: str = "basic",
        chunks_per_source: int | None = None,
    ) -> list[ExtractedPage]:
        try:
            raw = await self._client.extract(
                urls=urls,
                query=query,
                extract_depth=extract_depth,
                chunks_per_source=chunks_per_source,
            )
        except ClientError as exc:
            raise ServiceError(f"Web extract failed: {exc}") from exc

        pages: list[ExtractedPage] = []
        for item in raw.get("results", []):
            pages.append(ExtractedPage(
                url=item.get("url") or "",
                content=item.get("raw_content") or item.get("content") or "",
                images=item.get("images") or [],
            ))
        return pages

    async def crawl(
        self,
        url: str,
        instructions: str = "",
        max_depth: int = 1,
        limit: int = 20,
        select_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> CrawlResult:
        try:
            raw = await self._client.crawl(
                url=url,
                instructions=instructions,
                max_depth=max_depth,
                limit=limit,
                select_paths=select_paths,
                exclude_paths=exclude_paths,
            )
        except ClientError as exc:
            raise ServiceError(f"Web crawl failed for '{url}': {exc}") from exc

        pages: list[CrawlPage] = []
        for item in raw.get("results", []):
            pages.append(CrawlPage(
                url=item.get("url") or "",
                content=item.get("raw_content") or item.get("content") or "",
            ))
        return CrawlResult(root_url=url, pages_crawled=len(pages), pages=pages)

    async def map(self, url: str, instructions: str = "", limit: int = 50) -> MapResult:
        try:
            raw = await self._client.map(url=url, instructions=instructions, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"Web map failed for '{url}': {exc}") from exc

        urls = raw.get("urls", [])
        if not isinstance(urls, list):
            urls = []
        return MapResult(root_url=url, urls=urls)

    async def research(self, query: str, model: str = "auto") -> ResearchReport:
        try:
            raw = await self._research_client.research(query=query, model=model)
        except ClientError as exc:
            raise ServiceError(f"Web research failed: {exc}") from exc

        report_text = (
            raw.get("report")
            or raw.get("answer")
            or raw.get("content")
            or ""
        )
        citations: list[str] = []
        for src in raw.get("sources", raw.get("citations", [])):
            if isinstance(src, dict):
                citations.append(src.get("url") or str(src))
            elif isinstance(src, str):
                citations.append(src)

        return ResearchReport(
            query=query,
            model=model,
            report=report_text,
            citations=citations,
        )
