import asyncio
import time
import httpx


class ClientError(Exception):
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


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
    ) -> None:
        self._base_url = base_url
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._rate_limit_per_min = rate_limit_per_min

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

    async def get(self, path: str, params: dict | None = None) -> dict:
        await self._acquire_token()
        last_error: ClientError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.get(
                    path, params=params, headers=self._auth_headers()
                )
                if response.status_code in self._RETRYABLE_STATUSES and attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if response.is_error:
                    raise ClientError(response.status_code, response.text)
                return response.json()
            except ClientError as exc:
                last_error = exc
                if exc.status not in self._RETRYABLE_STATUSES or attempt >= self._max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise last_error  # type: ignore[misc]

    async def post(self, path: str, json: dict | None = None) -> dict:
        await self._acquire_token()
        last_error: ClientError | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    path, json=json, headers=self._auth_headers()
                )
                if response.status_code in self._RETRYABLE_STATUSES and attempt < self._max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                if response.is_error:
                    raise ClientError(response.status_code, response.text)
                return response.json()
            except ClientError as exc:
                last_error = exc
                if exc.status not in self._RETRYABLE_STATUSES or attempt >= self._max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)
        raise last_error  # type: ignore[misc]

    async def close(self) -> None:
        await self._client.aclose()
