from config import settings
from clients.base_http_client import BaseHttpClient


class FredClient(BaseHttpClient):
    """FRED (Federal Reserve Economic Data) client — returns raw dicts, no business logic.
    API key goes in query params; no Authorization header needed."""

    def __init__(self) -> None:
        super().__init__(
            base_url=settings.fred_base_url,
            api_key="",
            timeout_s=settings.fred_timeout_s,
            max_retries=settings.fred_max_retries,
            rate_limit_per_min=settings.fred_rate_limit_per_min,
        )
        self._api_key = settings.fred_api_key

    def _auth_headers(self) -> dict:
        return {}

    async def get_observations(self, series_id: str, limit: int = 10) -> dict:
        """GET /series/observations — most-recent N observations, descending."""
        return await self.get("/series/observations", params={
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        })

    async def get_series_info(self, series_id: str) -> dict:
        """GET /series — series metadata (title, units, frequency)."""
        return await self.get("/series", params={
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
        })

    async def search_series(
        self,
        search_text: str,
        limit: int = 20,
        order_by: str = "search_rank",
        sort_order: str = "desc",
    ) -> dict:
        """GET /series/search — full-text search across all FRED series."""
        return await self.get("/series/search", params={
            "search_text": search_text,
            "api_key": self._api_key,
            "file_type": "json",
            "limit": limit,
            "order_by": order_by,
            "sort_order": sort_order,
        })

    async def get_observations_extended(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        limit: int = 100,
        units: str = "lin",
        frequency: str | None = None,
        sort_order: str = "desc",
    ) -> dict:
        """GET /series/observations — with optional date range and unit/frequency transforms."""
        params: dict = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": sort_order,
            "limit": limit,
            "units": units,
        }
        if observation_start:
            params["observation_start"] = observation_start
        if observation_end:
            params["observation_end"] = observation_end
        if frequency:
            params["frequency"] = frequency
        return await self.get("/series/observations", params=params)

    async def get_releases(self, limit: int = 50) -> dict:
        """GET /releases — all economic data releases."""
        return await self.get("/releases", params={
            "api_key": self._api_key,
            "file_type": "json",
            "limit": limit,
            "order_by": "release_id",
            "sort_order": "asc",
        })

    async def get_release(self, release_id: int) -> dict:
        """GET /release — a single economic data release."""
        return await self.get("/release", params={
            "release_id": release_id,
            "api_key": self._api_key,
            "file_type": "json",
        })

    async def get_release_series(self, release_id: int, limit: int = 50) -> dict:
        """GET /release/series — series belonging to a release, sorted by popularity."""
        return await self.get("/release/series", params={
            "release_id": release_id,
            "api_key": self._api_key,
            "file_type": "json",
            "limit": limit,
            "order_by": "popularity",
            "sort_order": "desc",
        })

    async def get_category(self, category_id: int = 0) -> dict:
        """GET /category — category metadata."""
        return await self.get("/category", params={
            "category_id": category_id,
            "api_key": self._api_key,
            "file_type": "json",
        })

    async def get_category_children(self, category_id: int = 0) -> dict:
        """GET /category/children — immediate child categories."""
        return await self.get("/category/children", params={
            "category_id": category_id,
            "api_key": self._api_key,
            "file_type": "json",
        })

    async def get_category_series(self, category_id: int, limit: int = 50) -> dict:
        """GET /category/series — series in a category, sorted by popularity."""
        return await self.get("/category/series", params={
            "category_id": category_id,
            "api_key": self._api_key,
            "file_type": "json",
            "limit": limit,
            "order_by": "popularity",
            "sort_order": "desc",
        })

    async def get_tags_series(self, tag_names: str, limit: int = 20) -> dict:
        """GET /tags/series — series matching all specified tags, sorted by popularity."""
        return await self.get("/tags/series", params={
            "tag_names": tag_names,
            "api_key": self._api_key,
            "file_type": "json",
            "limit": limit,
            "order_by": "popularity",
            "sort_order": "desc",
        })

    async def get_series_updates(self, limit: int = 50, filter_value: str = "macro") -> dict:
        """GET /series/updates — recently updated series."""
        return await self.get("/series/updates", params={
            "api_key": self._api_key,
            "file_type": "json",
            "limit": limit,
            "filter_value": filter_value,
        })


_shared_client: FredClient | None = None


def get_fred_client() -> FredClient:
    """Return a process-wide shared FredClient so all FRED tools share one httpx
    connection pool and one token-bucket rate limiter (FRED's real limit is global,
    ~120 req/min). Per-tool clients would each get their own bucket and collectively
    blow past the limit under parallel domain agents."""
    global _shared_client
    if _shared_client is None:
        _shared_client = FredClient()
    return _shared_client
