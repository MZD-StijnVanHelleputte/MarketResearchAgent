from reports.chart_renderer import render_chart, should_chart

_MAX_ROWS = 10


def _price_series(n: int) -> dict:
    return {
        "kind": "table",
        "title": "Copper price",
        "columns": ["date", "value (usd)"],
        "rows": [[f"2026-01-{i:02d}", str(100 + i)] for i in range(1, n + 1)],
    }


def _ohlcv_series(n: int) -> dict:
    return {
        "kind": "table",
        "title": "CAT 5y",
        "columns": ["date", "open", "high", "low", "close", "volume"],
        "rows": [
            [f"2026-01-{i:02d}", "350", "355", "348", "352", "1000000"]
            for i in range(1, n + 1)
        ],
    }


def _categorical_series(n: int) -> dict:
    return {
        "kind": "table",
        "title": "Revenue by segment",
        "columns": ["segment", "revenue"],
        "rows": [[f"segment-{i}", str(1000 + i)] for i in range(n)],
    }


def _non_numeric_table(n: int) -> dict:
    return {
        "kind": "table",
        "title": "IDs",
        "columns": ["id", "label"],
        "rows": [[f"id-{i}", f"label-{i}"] for i in range(n)],
    }


def test_should_chart_false_below_row_threshold():
    assert should_chart(_price_series(5), _MAX_ROWS) is False


def test_should_chart_true_for_date_value_series_over_threshold():
    assert should_chart(_price_series(_MAX_ROWS + 5), _MAX_ROWS) is True


def test_should_chart_false_for_non_numeric_table():
    assert should_chart(_non_numeric_table(_MAX_ROWS + 5), _MAX_ROWS) is False


def test_should_chart_false_for_non_table_kind():
    assert should_chart({"kind": "list", "items": []}, _MAX_ROWS) is False


def test_render_chart_line_for_date_series_returns_png_bytes():
    png = render_chart(_price_series(_MAX_ROWS + 5))
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_chart_handles_ohlcv_multi_column():
    png = render_chart(_ohlcv_series(_MAX_ROWS + 5))
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_chart_bar_fallback_for_categorical_data():
    png = render_chart(_categorical_series(_MAX_ROWS + 5))
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_chart_returns_none_on_malformed_dataset():
    ds = {"kind": "table", "title": "Broken", "rows": [["2026-01-01", "100"]]}  # missing "columns"
    assert render_chart(ds) is None


def test_render_chart_tolerates_ragged_rows():
    ds = {
        "kind": "table",
        "title": "Ragged",
        "columns": ["date", "value"],
        "rows": [["2026-01-01", "100"], ["2026-01-02"], ["2026-01-03", "102", "extra"]],
    }
    # Ragged rows are dropped by length-filtering inside _render; with only one
    # well-formed row left this should still not raise.
    render_chart(ds)
