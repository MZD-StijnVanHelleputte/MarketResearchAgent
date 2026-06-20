import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from clients.base_http_client import BaseHttpClient, ClientError


def _make_response(status_code: int, json_body: dict | None = None, text: str = "") -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.is_error = status_code >= 400
    resp.text = text
    resp.json.return_value = json_body or {}
    return resp


@pytest.mark.asyncio
async def test_get_success():
    client = BaseHttpClient(base_url="https://example.com", rate_limit_per_min=1000)
    mock_resp = _make_response(200, {"status": "ok"})

    with patch.object(client._client, "get", new=AsyncMock(return_value=mock_resp)):
        result = await client.get("/test")

    assert result == {"status": "ok"}


@pytest.mark.asyncio
async def test_get_retries_on_429_then_succeeds():
    client = BaseHttpClient(base_url="https://example.com", max_retries=2, rate_limit_per_min=1000)
    fail = _make_response(429, text="rate limited")
    ok = _make_response(200, {"data": "good"})

    call_count = 0

    async def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return fail if call_count < 2 else ok

    with patch.object(client._client, "get", new=AsyncMock(side_effect=side_effect)):
        with patch("clients.base_http_client.asyncio.sleep", new=AsyncMock()):
            result = await client.get("/test")

    assert result == {"data": "good"}
    assert call_count == 2


@pytest.mark.asyncio
async def test_get_raises_client_error_on_404():
    client = BaseHttpClient(base_url="https://example.com", rate_limit_per_min=1000)
    mock_resp = _make_response(404, text="not found")

    with patch.object(client._client, "get", new=AsyncMock(return_value=mock_resp)):
        with pytest.raises(ClientError) as exc_info:
            await client.get("/missing")

    assert exc_info.value.status == 404


@pytest.mark.asyncio
async def test_get_exhausts_retries_and_raises():
    client = BaseHttpClient(base_url="https://example.com", max_retries=1, rate_limit_per_min=1000)
    fail = _make_response(503, text="unavailable")

    with patch.object(client._client, "get", new=AsyncMock(return_value=fail)):
        with patch("clients.base_http_client.asyncio.sleep", new=AsyncMock()):
            with pytest.raises(ClientError) as exc_info:
                await client.get("/test")

    assert exc_info.value.status == 503


@pytest.mark.asyncio
async def test_rate_limit_burst_produces_delay():
    """A burst of 12 calls at rate_limit_per_min=10 must cause at least one asyncio.sleep."""
    sleep_calls: list[float] = []

    async def capture_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    client = BaseHttpClient(base_url="https://example.com", rate_limit_per_min=10)
    ok = _make_response(200, {"ok": True})

    with patch.object(client._client, "get", new=AsyncMock(return_value=ok)):
        with patch("clients.base_http_client.asyncio.sleep", new=AsyncMock(side_effect=capture_sleep)):
            for _ in range(12):
                await client.get("/test")

    # With only 10 tokens/min, 12 calls must trigger at least one rate-limit sleep
    assert len(sleep_calls) >= 1
    # All sleep durations should be positive
    assert all(s > 0 for s in sleep_calls)
