import asyncio
import random
import time
from typing import Any, Callable
import httpx


class ClientError(Exception):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


# Permanent client-side errors (bad request, not found, unprocessable). These can never
# succeed on retry or via argument repair, so callers should fail fast on them. 429 is
# deliberately excluded — it's rate limiting and is retryable.
PERMANENT_STATUSES = {400, 404, 422}

# Synthetic status used when a transport-level error (timeout, connection reset) exhausts
# all retries and is surfaced as a ClientError for uniform downstream handling.
TRANSPORT_ERROR_STATUS = 599


class BaseHttpClient:
    """Async HTTP client with retry, exponential backoff, and token-bucket rate limiting."""

    _RETRYABLE_STATUSES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout_s: int = 10,
        max_retries: int = 3,
        rate_limit_per_min: int = 60,
        backoff_cap_s: float = 30.0,
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._rate_limit_per_min = rate_limit_per_min
        self._backoff_cap_s = backoff_cap_s

        self._client = httpx.AsyncClient(base_url=base_url, timeout=timeout_s)

        # Token-bucket state
        self._rate_lock = asyncio.Lock()
        self._bucket_tokens = float(rate_limit_per_min)
        self._bucket_last_refill = time.monotonic()

    async def _acquire_token(self) -> None:
        async with self._rate_lock:
            now = time.monotonic()
            elapsed = now - self._bucket_last_refill
            refill = elapsed * (self._rate_limit_per_min / 60.0)
            self._bucket_tokens = min(
                float(self._rate_limit_per_min),
                self._bucket_tokens + refill,
            )
            self._bucket_last_refill = now

            if self._bucket_tokens < 1:
                wait = (1 - self._bucket_tokens) / (self._rate_limit_per_min / 60.0)
                await asyncio.sleep(wait)
                self._bucket_tokens = 0.0
            else:
                self._bucket_tokens -= 1

    def _auth_headers(self) -> dict:
        if self._api_key:
            return {"Authorization": f"Bearer {self._api_key}"}
        return {}

    async def _backoff(self, attempt: int) -> None:
        """Exponential backoff with jitter so parallel clients don't retry in lockstep.
        Capped at backoff_cap_s so rate-limited clients fail fast to their fallback."""
        await asyncio.sleep(min(2 ** attempt, self._backoff_cap_s) + random.uniform(0, 0.5))

    async def get(self, path: str, params: dict | None = None) -> dict:
        return await self._with_retries(
            lambda: self._client.get(path, params=params, headers=self._auth_headers())
        )

    async def post(self, path: str, json: dict | None = None) -> dict:
        return await self._with_retries(
            lambda: self._client.post(path, json=json, headers=self._auth_headers())
        )

    async def get_bytes(self, path: str, params: dict | None = None) -> bytes:
        """Like get(), but returns the raw response body instead of parsing JSON —
        for non-JSON resources such as filing documents (HTML/PDF exhibits)."""
        return await self._with_retries(
            lambda: self._client.get(path, params=params, headers=self._auth_headers()),
            parse=lambda r: r.content,
        )

    async def _with_retries(
        self, send, parse: Callable[[httpx.Response], Any] = lambda r: r.json()
    ) -> Any:
        """Run an httpx request callable with rate limiting, status-based retries, and
        transport-level (timeout/connection) retries, all using jittered backoff."""
        await self._acquire_token()
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await send()
                if response.status_code in self._RETRYABLE_STATUSES and attempt < self._max_retries:
                    await self._backoff(attempt)
                    continue
                if response.is_error:
                    raise ClientError(response.status_code, response.text)
                return parse(response)
            except ClientError as exc:
                last_error = exc
                if exc.status not in self._RETRYABLE_STATUSES or attempt >= self._max_retries:
                    raise
                await self._backoff(attempt)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                # Transient transport-level failure (timeout, connection reset, protocol
                # error). FRED is intermittently slow, so retry rather than fail instantly.
                last_error = exc
                if attempt >= self._max_retries:
                    raise ClientError(TRANSPORT_ERROR_STATUS, str(exc)) from exc
                await self._backoff(attempt)
        raise last_error  # type: ignore[misc]

    async def close(self) -> None:
        await self._client.aclose()
