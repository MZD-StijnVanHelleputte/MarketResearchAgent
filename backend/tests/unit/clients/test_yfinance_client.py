import pytest
from unittest.mock import MagicMock, patch
from clients.yfinance_client import YFinanceClient


def _make_fast_info(last_price=42.5, market_cap=15e9, currency="USD"):
    info = MagicMock()
    info.last_price = last_price
    info.market_cap = market_cap
    info.currency = currency
    return info


def _make_history_df():
    import pandas as pd
    from datetime import datetime
    data = {
        "Open": [40.0, 41.0],
        "High": [43.0, 44.0],
        "Low": [39.0, 40.0],
        "Close": [42.0, 43.0],
        "Volume": [1000000, 1100000],
    }
    index = pd.DatetimeIndex([datetime(2026, 1, 2), datetime(2026, 1, 3)])
    return pd.DataFrame(data, index=index)


@pytest.mark.asyncio
async def test_get_price_returns_dict():
    client = YFinanceClient()
    mock_ticker = MagicMock()
    mock_ticker.fast_info = _make_fast_info()

    with patch("clients.yfinance_client.asyncio.to_thread") as mock_thread:
        mock_thread.side_effect = lambda fn: fn()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await client.get_price("FCX")

    assert result["symbol"] == "FCX"
    assert result["price"] == 42.5
    assert result["currency"] == "USD"
    assert result["market_cap"] == 15e9
    assert "date" in result


@pytest.mark.asyncio
async def test_get_history_returns_list():
    client = YFinanceClient()
    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _make_history_df()

    with patch("clients.yfinance_client.asyncio.to_thread") as mock_thread:
        mock_thread.side_effect = lambda fn: fn()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await client.get_history("FCX", period="1mo")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["date"] == "2026-01-02"
    assert result[0]["close"] == 42.0


def _make_financials_df():
    import pandas as pd
    from datetime import datetime
    data = {
        datetime(2025, 12, 31): {"Total Revenue": 1000.0, "Net Income": 100.0},
        datetime(2024, 12, 31): {"Total Revenue": 900.0, "Net Income": 80.0},
    }
    return pd.DataFrame(data)


@pytest.mark.asyncio
async def test_get_financials_returns_list_annual():
    client = YFinanceClient()
    mock_ticker = MagicMock()
    mock_ticker.financials = _make_financials_df()

    with patch("clients.yfinance_client.asyncio.to_thread") as mock_thread:
        mock_thread.side_effect = lambda fn: fn()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await client.get_financials("CAT", period="annual")

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["date"] == "2025-12-31"
    assert result[0]["Total Revenue"] == 1000.0
    assert result[0]["Net Income"] == 100.0


def _make_cashflow_df():
    import pandas as pd
    from datetime import datetime
    data = {
        datetime(2025, 12, 31): {"Capital Expenditure": -2000.0},
        datetime(2024, 12, 31): {"Capital Expenditure": -1800.0},
    }
    return pd.DataFrame(data)


@pytest.mark.asyncio
async def test_get_company_overview_returns_dict():
    client = YFinanceClient()
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "longName": "Caterpillar Inc", "currentPrice": 380.0, "marketCap": 150e9,
        "totalRevenue": 67e9, "netIncomeToCommon": 6e9, "trailingPE": 15.2,
        "financialCurrency": "USD", "industry": "Machinery",
    }
    mock_ticker.cashflow = _make_cashflow_df()

    with patch("clients.yfinance_client.asyncio.to_thread") as mock_thread:
        mock_thread.side_effect = lambda fn: fn()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await client.get_company_overview("CAT")

    assert result["symbol"] == "CAT"
    assert result["name"] == "Caterpillar Inc"
    assert result["price"] == 380.0
    assert result["market_cap"] == 150e9
    assert result["revenue"] == 67e9
    assert result["net_income"] == 6e9
    assert result["pe_ratio"] == 15.2
    assert result["capex"] == -2000.0  # latest period from cashflow
    assert result["currency"] == "USD"
    assert result["industry"] == "Machinery"


@pytest.mark.asyncio
async def test_get_company_overview_tolerates_missing_fields():
    client = YFinanceClient()
    mock_ticker = MagicMock()
    mock_ticker.info = {}
    mock_ticker.cashflow = None

    with patch("clients.yfinance_client.asyncio.to_thread") as mock_thread:
        mock_thread.side_effect = lambda fn: fn()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await client.get_company_overview("XYZ")

    assert result["name"] is None
    assert result["revenue"] is None
    assert result["capex"] is None
    assert result["currency"] == "USD"  # default


@pytest.mark.asyncio
async def test_get_financials_uses_quarterly_financials():
    client = YFinanceClient()
    mock_ticker = MagicMock()
    mock_ticker.quarterly_financials = _make_financials_df()

    with patch("clients.yfinance_client.asyncio.to_thread") as mock_thread:
        mock_thread.side_effect = lambda fn: fn()
        with patch("yfinance.Ticker", return_value=mock_ticker):
            result = await client.get_financials("CAT", period="quarterly")

    assert len(result) == 2
    assert result[0]["Total Revenue"] == 1000.0
