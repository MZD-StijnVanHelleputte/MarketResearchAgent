from dataclasses import dataclass
from clients.newsapi_client import NewsApiClient
from clients.base_http_client import ClientError


class ServiceError(Exception):
    pass


@dataclass
class NewsArticle:
    title: str
    description: str | None
    url: str
    published_at: str
    source: str


@dataclass
class NewsSource:
    id: str
    name: str
    description: str | None
    url: str
    category: str | None
    language: str | None
    country: str | None


def _parse_article(a: dict) -> NewsArticle:
    return NewsArticle(
        title=a.get("title") or "",
        description=a.get("description"),
        url=a.get("url") or "",
        published_at=a.get("publishedAt") or "",
        source=(a.get("source") or {}).get("name") or "",
    )


class NewsService:
    """Business logic wrapper over NewsApiClient."""

    def __init__(self, client: NewsApiClient | None = None) -> None:
        self._client = client or NewsApiClient()

    async def search(
        self,
        query: str,
        language: str = "en",
        page_size: int = 5,
        from_date: str | None = None,
    ) -> list[NewsArticle]:
        try:
            raw = await self._client.search(
                query=query,
                language=language,
                page_size=page_size,
                from_date=from_date,
            )
        except ClientError as exc:
            raise ServiceError(f"NewsAPI request failed: {exc}") from exc

        if raw.get("status") != "ok":
            raise ServiceError(f"NewsAPI returned non-ok status: {raw.get('status')}")

        return [_parse_article(a) for a in raw.get("articles", [])]

    async def top_headlines(
        self,
        query: str | None = None,
        country: str | None = None,
        category: str | None = None,
        sources: str | None = None,
        page_size: int = 5,
        page: int = 1,
    ) -> list[NewsArticle]:
        try:
            raw = await self._client.top_headlines(
                query=query,
                country=country,
                category=category,
                sources=sources,
                page_size=page_size,
                page=page,
            )
        except ClientError as exc:
            raise ServiceError(f"NewsAPI request failed: {exc}") from exc

        if raw.get("status") != "ok":
            raise ServiceError(f"NewsAPI returned non-ok status: {raw.get('status')}")

        return [_parse_article(a) for a in raw.get("articles", [])]

    async def list_sources(
        self,
        category: str | None = None,
        language: str | None = None,
        country: str | None = None,
    ) -> list[NewsSource]:
        try:
            raw = await self._client.sources(category=category, language=language, country=country)
        except ClientError as exc:
            raise ServiceError(f"NewsAPI request failed: {exc}") from exc

        if raw.get("status") != "ok":
            raise ServiceError(f"NewsAPI returned non-ok status: {raw.get('status')}")

        return [
            NewsSource(
                id=s.get("id") or "",
                name=s.get("name") or "",
                description=s.get("description"),
                url=s.get("url") or "",
                category=s.get("category"),
                language=s.get("language"),
                country=s.get("country"),
            )
            for s in raw.get("sources", [])
        ]
