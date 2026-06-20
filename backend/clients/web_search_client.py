from config import settings
from clients.base_http_client import BaseHttpClient


class WebSearchClient(BaseHttpClient):
    """Tavily client — returns dicts, no business logic.
    Tavily authenticates via the `Authorization: Bearer <key>` header."""

    def __init__(self) -> None:
        super().__init__(
            base_url=settings.tavily_base_url,
            api_key=settings.tavily_api_key,
            timeout_s=settings.tavily_timeout_s,
            rate_limit_per_min=settings.tavily_rate_limit_per_min,
        )

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
    ) -> dict:
        """POST /search — web search with configurable depth, topic, and filters."""
        body: dict = {
            "query": query,
            "max_results": max_results,
            "search_depth": search_depth,
            "topic": topic,
            "include_answer": include_answer,
            "include_raw_content": include_raw_content,
        }
        if time_range:
            body["time_range"] = time_range
        if include_domains:
            body["include_domains"] = include_domains
        if exclude_domains:
            body["exclude_domains"] = exclude_domains
        return await self.post("/search", json=body)

    async def extract(
        self,
        urls: list[str],
        query: str = "",
        extract_depth: str = "basic",
        chunks_per_source: int | None = None,
    ) -> dict:
        """POST /extract — clean content extraction from specific URLs (up to 20)."""
        body: dict = {
            "urls": urls,
            "extract_depth": extract_depth,
        }
        if query:
            body["query"] = query
        if chunks_per_source is not None and query:
            body["chunks_per_source"] = chunks_per_source
        return await self.post("/extract", json=body)

    async def crawl(
        self,
        url: str,
        instructions: str = "",
        max_depth: int = 1,
        limit: int = 20,
        select_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
    ) -> dict:
        """POST /crawl — graph-based multi-page website extraction."""
        body: dict = {
            "url": url,
            "max_depth": max_depth,
            "limit": limit,
        }
        if instructions:
            body["instructions"] = instructions
        if select_paths:
            body["select_paths"] = select_paths
        if exclude_paths:
            body["exclude_paths"] = exclude_paths
        return await self.post("/crawl", json=body)

    async def map(
        self,
        url: str,
        instructions: str = "",
        limit: int = 50,
    ) -> dict:
        """POST /map — URL discovery without content extraction."""
        body: dict = {
            "url": url,
            "limit": limit,
        }
        if instructions:
            body["instructions"] = instructions
        return await self.post("/map", json=body)


class WebResearchClient(WebSearchClient):
    """Tavily /research client — same auth, longer timeout for 30–120 s responses."""

    def __init__(self) -> None:
        # Call BaseHttpClient.__init__ directly to set a longer timeout
        BaseHttpClient.__init__(
            self,
            base_url=settings.tavily_base_url,
            api_key=settings.tavily_api_key,
            timeout_s=settings.tavily_research_timeout_s,
            rate_limit_per_min=settings.tavily_rate_limit_per_min,
        )

    async def research(self, query: str, model: str = "auto") -> dict:
        """POST /research — AI-synthesized multi-source research report."""
        body: dict = {
            "query": query,
            "model": model,
        }
        return await self.post("/research", json=body)
