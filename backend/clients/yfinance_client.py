import asyncio
from datetime import date


class YFinanceClient:
    """Async wrapper around the yfinance library (synchronous).
    All calls run in a thread pool to avoid blocking the event loop."""

    async def get_price(self, ticker: str) -> dict:
        """Return latest price info for a ticker symbol."""
        def _fetch():
            import yfinance as yf
            info = yf.Ticker(ticker).fast_info
            return {
                "symbol": ticker.upper(),
                "price": float(info.last_price) if info.last_price else None,
                "currency": getattr(info, "currency", "USD"),
                "market_cap": float(info.market_cap) if info.market_cap else None,
                "date": date.today().isoformat(),
            }

        return await asyncio.to_thread(_fetch)

    async def get_financials(self, ticker: str, period: str = "annual") -> list[dict]:
        """Return income-statement line items per period. period: annual | quarterly."""
        def _fetch():
            import yfinance as yf
            t = yf.Ticker(ticker)
            df = t.financials if period == "annual" else t.quarterly_financials
            rows = []
            for col in df.columns:
                row: dict = {"date": col.date().isoformat() if hasattr(col, "date") else str(col)}
                for line_item, value in df[col].items():
                    if value is None or value != value:  # filter NaN
                        continue
                    row[str(line_item)] = round(float(value), 2)
                rows.append(row)
            return rows

        return await asyncio.to_thread(_fetch)

    async def get_history(self, ticker: str, period: str = "3mo") -> list[dict]:
        """Return OHLCV history for a ticker. period: 1mo | 3mo | 6mo | 1y | 2y."""
        def _fetch():
            import yfinance as yf
            df = yf.Ticker(ticker).history(period=period)
            rows = []
            for ts, row in df.iterrows():
                rows.append({
                    "date": ts.date().isoformat(),
                    "open": round(float(row["Open"]), 4),
                    "high": round(float(row["High"]), 4),
                    "low": round(float(row["Low"]), 4),
                    "close": round(float(row["Close"]), 4),
                    "volume": int(row["Volume"]),
                })
            return rows

        return await asyncio.to_thread(_fetch)
