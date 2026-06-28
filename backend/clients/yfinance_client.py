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

    async def get_company_overview(self, ticker: str) -> dict:
        """Return a one-shot financial summary (name, price, market cap, revenue, net income,
        capex, P/E, currency, industry) from yfinance — the free, no-rate-limit fallback for
        FMP's get_company_financials. Missing fields come back as None."""
        def _fetch():
            import yfinance as yf
            t = yf.Ticker(ticker)
            info = t.info or {}

            capex = None
            try:
                cf = t.cashflow
                if cf is not None and not cf.empty and "Capital Expenditure" in cf.index:
                    val = cf.loc["Capital Expenditure"].iloc[0]  # latest period
                    if val is not None and val == val:  # filter NaN
                        capex = round(float(val), 2)
            except Exception:
                pass  # capex is best-effort; profile fields are what matter

            return {
                "symbol": ticker.upper(),
                "name": info.get("longName") or info.get("shortName"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "revenue": info.get("totalRevenue"),
                "net_income": info.get("netIncomeToCommon"),
                "capex": capex,
                "pe_ratio": info.get("trailingPE"),
                "currency": info.get("financialCurrency") or info.get("currency") or "USD",
                "industry": info.get("industry"),
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
