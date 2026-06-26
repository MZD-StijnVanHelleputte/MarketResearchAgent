from config import settings
from clients.base_http_client import BaseHttpClient

_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


class EdgarClient(BaseHttpClient):
    """SEC EDGAR client. No API key — User-Agent header required by SEC policy.

    Covers two EDGAR surfaces: the full-text search API (efts.sec.gov, used by
    search_filings) and the submissions/Archives APIs (data.sec.gov, www.sec.gov,
    used to resolve a ticker to a CIK and fetch filing documents). httpx ignores
    base_url when given an absolute URL, so a single client/rate-limit bucket can
    serve both — important since SEC enforces one combined ~10 req/sec budget
    across all of these hosts.
    """

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
        self._ticker_cik_cache: dict[str, str] | None = None

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

    async def get_cik_for_ticker(self, ticker: str) -> str | None:
        """Resolve a stock ticker to a 10-digit zero-padded CIK.

        Fetches the SEC's static ticker->CIK mapping once and caches it for the
        client's lifetime (the file covers ~10k tickers and changes infrequently).
        """
        if self._ticker_cik_cache is None:
            # Response shape: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
            raw = await self.get(_TICKER_MAP_URL)
            self._ticker_cik_cache = {
                str(row.get("ticker", "")).upper(): str(row.get("cik_str", "")).zfill(10)
                for row in raw.values()
                if isinstance(row, dict) and row.get("ticker")
            }
        return self._ticker_cik_cache.get(ticker.upper())

    async def get_submissions(self, cik10: str) -> dict:
        """GET the filing submissions history (recent filings + metadata) for a CIK."""
        return await self.get(f"https://data.sec.gov/submissions/CIK{cik10}.json")

    async def get_document_bytes(self, url: str) -> bytes:
        """Fetch the raw bytes of a filing document/exhibit (HTML or PDF)."""
        return await self.get_bytes(url)


_shared_client: EdgarClient | None = None


def get_edgar_client() -> EdgarClient:
    """Return a process-wide shared EdgarClient so all EDGAR tools share one httpx
    connection pool and one token-bucket rate limiter (SEC's real limit is a combined
    ~10 req/sec across efts.sec.gov/data.sec.gov/www.sec.gov). Per-tool clients would
    each get their own bucket and collectively blow past the limit under parallel
    domain agents."""
    global _shared_client
    if _shared_client is None:
        _shared_client = EdgarClient()
    return _shared_client
