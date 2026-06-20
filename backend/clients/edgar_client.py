from config import settings
from clients.base_http_client import BaseHttpClient


class EdgarClient(BaseHttpClient):
    """SEC EDGAR full-text search client. No API key — User-Agent header required by SEC policy."""

    def __init__(self) -> None:
        super().__init__(
            base_url=settings.edgar_base_url,
            api_key="",
            timeout_s=settings.edgar_timeout_s,
            max_retries=settings.edgar_max_retries,
            rate_limit_per_min=settings.edgar_rate_limit_per_min,
        )
        # Contact email stored in sec_edgar_api_key field; required by SEC fair-access policy
        self._contact = settings.sec_edgar_api_key or "contact@example.com"

    def _auth_headers(self) -> dict:
        return {"User-Agent": f"KomatsuIntel/1.0 ({self._contact})"}

    async def search_filings(
        self,
        query: str,
        forms: str = "10-K,10-Q,8-K",
        limit: int = 5,
    ) -> dict:
        """Full-text search across SEC filings via EDGAR EFTS API.

        Valid request params are q, forms, dateRange, startdt, enddt, from, size; the
        endpoint returns up to ~10 hits per page (paginate via `from`). `limit` is
        applied client-side by the service.
        """
        params = {
            "q": f'"{query}"',
            "forms": forms,
            "from": 0,
        }
        return await self.get("/LATEST/search-index", params=params)
