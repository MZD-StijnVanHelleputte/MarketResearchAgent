"""Diagnostic smoke test for the collection-phase external APIs.

Probes each provider on the real keys in .env and prints the HTTP status +
a truncated body, so per-tool failures (currently only logged to stdout as
warnings and never persisted) become a single readable report.

For the two endpoints being migrated (Alpha Vantage commodities, FMP /stable/)
and for SEC EDGAR, it probes BOTH the current and the proposed-fixed shapes so
we can confirm the exact response field names before rewriting the parsers.

Run from the backend/ directory:

    python -m scripts.smoke_test_apis
"""
import asyncio
import json

import httpx

from config import settings

TRUNC = 600


def show(label: str, status, body) -> None:
    if isinstance(body, (dict, list)):
        body = json.dumps(body, default=str)
    body = str(body)
    if len(body) > TRUNC:
        body = body[:TRUNC] + " …"
    print(f"\n[{label}] status={status}\n  {body}")


async def probe_get(client: httpx.AsyncClient, label, url, params=None, headers=None):
    try:
        r = await client.get(url, params=params, headers=headers)
        body = r.json() if "json" in r.headers.get("content-type", "") else r.text
        show(label, r.status_code, body)
    except Exception as exc:  # noqa: BLE001
        show(label, "EXC", f"{type(exc).__name__}: {exc}")


async def probe_post(client: httpx.AsyncClient, label, url, json_body=None, headers=None):
    try:
        r = await client.post(url, json=json_body, headers=headers)
        body = r.json() if "json" in r.headers.get("content-type", "") else r.text
        show(label, r.status_code, body)
    except Exception as exc:  # noqa: BLE001
        show(label, "EXC", f"{type(exc).__name__}: {exc}")


async def main() -> None:
    ua = f"KomatsuIntel/1.0 ({settings.sec_edgar_api_key or 'contact@example.com'})"
    async with httpx.AsyncClient(timeout=20) as c:
        print("=" * 70)
        print("NEWSAPI")
        await probe_get(
            c, "newsapi /v2/everything",
            "https://newsapi.org/v2/everything",
            params={"q": "copper mining", "pageSize": 2, "sortBy": "publishedAt",
                    "apiKey": settings.newsapi_api_key},
        )

        print("=" * 70)
        print("TAVILY  (body api_key vs Bearer header)")
        await probe_post(
            c, "tavily body api_key",
            "https://api.tavily.com/search",
            json_body={"api_key": settings.tavily_api_key, "query": "copper price", "max_results": 2},
        )
        await probe_post(
            c, "tavily Bearer header",
            "https://api.tavily.com/search",
            json_body={"query": "copper price", "max_results": 2},
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
        )

        print("=" * 70)
        print("SEC EDGAR  (current bogus param vs fixed)")
        await probe_get(
            c, "edgar CURRENT (hits.hits.total.value)",
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": '"copper mining"', "forms": "10-K", "hits.hits.total.value": 5},
            headers={"User-Agent": ua},
        )
        await probe_get(
            c, "edgar FIXED (from/size)",
            "https://efts.sec.gov/LATEST/search-index",
            params={"q": '"copper mining"', "forms": "10-K", "from": 0},
            headers={"User-Agent": ua},
        )

        print("=" * 70)
        print("FMP  (legacy v3 vs stable)")
        await probe_get(
            c, "fmp CURRENT /api/v3/profile/CAT",
            "https://financialmodelingprep.com/api/v3/profile/CAT",
            params={"apikey": settings.fmp_api_key},
        )
        await probe_get(
            c, "fmp STABLE /stable/profile?symbol=CAT",
            "https://financialmodelingprep.com/stable/profile",
            params={"symbol": "CAT", "apikey": settings.fmp_api_key},
        )
        await probe_get(
            c, "fmp STABLE /stable/key-metrics-ttm?symbol=CAT",
            "https://financialmodelingprep.com/stable/key-metrics-ttm",
            params={"symbol": "CAT", "apikey": settings.fmp_api_key},
        )

        print("=" * 70)
        print("ALPHA VANTAGE  (GLOBAL_QUOTE on commodity vs commodity functions)")
        await probe_get(
            c, "av CURRENT GLOBAL_QUOTE symbol=COPPER",
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": "COPPER", "apikey": settings.alpha_vantage_api_key},
        )
        for fn in ("COPPER", "GOLD", "WTI"):
            await probe_get(
                c, f"av FIXED function={fn} interval=monthly",
                "https://www.alphavantage.co/query",
                params={"function": fn, "interval": "monthly", "apikey": settings.alpha_vantage_api_key},
            )

        print("=" * 70)
        print("FRED")
        await probe_get(
            c, "fred /series/observations DGS10",
            "https://api.stlouisfed.org/fred/series/observations",
            params={"series_id": "DGS10", "api_key": settings.fred_api_key,
                    "file_type": "json", "sort_order": "desc", "limit": 2},
        )

    print("=" * 70)
    print("YFINANCE")
    try:
        from clients.yfinance_client import YFinanceClient
        yf = YFinanceClient()
        show("yfinance get_price(CAT)", "ok", await yf.get_price("CAT"))
        hist = await yf.get_history("CAT", "1mo")
        show("yfinance get_history(CAT,1mo)", "ok", f"{len(hist)} rows; last={hist[-1] if hist else None}")
    except Exception as exc:  # noqa: BLE001
        show("yfinance", "EXC", f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
