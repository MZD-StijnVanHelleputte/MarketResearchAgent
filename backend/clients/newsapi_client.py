from config import settings
from clients.base_http_client import BaseHttpClient


class NewsApiClient(BaseHttpClient):
    """Raw NewsAPI client — returns dicts, no business logic."""

    def __init__(self) -> None:
        super().__init__(
            base_url=settings.newsapi_base_url,
            api_key="",  # NewsAPI uses apiKey query param, not Authorization header
            timeout_s=settings.newsapi_timeout_s,
            max_retries=settings.newsapi_max_retries,
            rate_limit_per_min=settings.newsapi_rate_limit_per_min,
        )
        self._api_key = settings.newsapi_api_key

    def _auth_headers(self) -> dict:
        return {}  # key travels as the apiKey query param

    async def search(
        self,
        query: str,
        language: str = "en",
        page_size: int = 5,
        from_date: str | None = None,
    ) -> dict:
        """Call /v2/everything and return the raw response dict."""
        params: dict = {
            "q": query,
            "language": language,
            "pageSize": page_size,
            "apiKey": self._api_key,
            "sortBy": "publishedAt",
        }
        if from_date:
            params["from"] = from_date
        return await self.get("/v2/everything", params=params)

    async def top_headlines(
        self,
        query: str | None = None,
        country: str | None = None,
        category: str | None = None,
        sources: str | None = None,
        page_size: int = 5,
        page: int = 1,
    ) -> dict:
        """Call /v2/top-headlines and return the raw response dict."""
        params: dict = {
            "pageSize": page_size,
            "page": page,
            "apiKey": self._api_key,
        }
        if query:
            params["q"] = query
        if country:
            params["country"] = country
        if category:
            params["category"] = category
        if sources:
            params["sources"] = sources
        return await self.get("/v2/top-headlines", params=params)

    async def sources(
        self,
        category: str | None = None,
        language: str | None = None,
        country: str | None = None,
    ) -> dict:
        """Call /v2/top-headlines/sources and return the raw response dict."""
        params: dict = {"apiKey": self._api_key}
        if category:
            params["category"] = category
        if language:
            params["language"] = language
        if country:
            params["country"] = country
        return await self.get("/v2/top-headlines/sources", params=params)
