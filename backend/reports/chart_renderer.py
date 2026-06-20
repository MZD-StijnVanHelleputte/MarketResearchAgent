"""Renders numeric table-kind datasets as static chart images for PDF embedding.

A dataset is charted instead of tabulated when it has more rows than the
configured table threshold AND at least one column that's mostly numeric.
Date-led tables (commodity/equity price history) become line charts, one
line per numeric column; everything else falls back to a bar chart using the
first column as category labels. Never raises — callers must fall back to a
table when `render_chart()` returns None.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

_NUMERIC_SAMPLE_SIZE = 20
_NUMERIC_MIN_RATIO = 0.8
_MAX_BARS = 50
_VOLUME_COLUMN_RE = re.compile(r"volume", re.IGNORECASE)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def should_chart(dataset: dict, max_rows: int) -> bool:
    """True if this table-kind dataset should be rendered as a chart instead."""
    if dataset.get("kind") != "table":
        return False
    rows = dataset.get("rows") or []
    columns = dataset.get("columns") or []
    if len(rows) <= max_rows or not columns:
        return False
    return bool(_numeric_column_indices(columns, rows))


def render_chart(dataset: dict) -> bytes | None:
    """Render *dataset* as a PNG. Returns None on any failure (never raises)."""
    try:
        return _render(dataset)
    except Exception as exc:
        logger.warning("chart_renderer: failed to render %r: %s", dataset.get("title"), exc)
        return None


def _render(dataset: dict) -> bytes:
    columns = dataset["columns"]
    rows = [r for r in dataset["rows"] if len(r) == len(columns)]
    numeric_idx = _numeric_column_indices(columns, rows)
    date_col = 0 if _is_date_column(columns, rows) else None

    fig, ax = plt.subplots(figsize=(6.2, 3.2), dpi=150)
    if date_col is not None:
        _render_line(ax, columns, rows, date_col, numeric_idx)
    else:
        _render_bar(ax, columns, rows, numeric_idx[0])

    ax.set_title(dataset.get("title", ""), fontsize=9)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _render_line(ax, columns: list, rows: list[list], date_col: int, numeric_idx: list[int]) -> None:
    x = [r[date_col] for r in rows]
    plot_idx = [i for i in numeric_idx if i != date_col]
    # Drop a volume-like column when other price/value columns exist — its scale
    # would otherwise dwarf everything else on a single shared axis.
    if len(plot_idx) > 1:
        plot_idx = [i for i in plot_idx if not _VOLUME_COLUMN_RE.search(str(columns[i]))] or plot_idx

    for idx in plot_idx:
        y = _safe_floats(r[idx] for r in rows)
        ax.plot(x, y, label=str(columns[idx]), linewidth=1.2)

    ax.set_xlabel("Date")
    _thin_xticks(ax, x)
    if len(plot_idx) > 1:
        ax.legend(fontsize=7)


def _render_bar(ax, columns: list, rows: list[list], value_idx: int) -> None:
    sample = rows[:_MAX_BARS]
    labels = [str(r[0]) for r in sample]
    y = _safe_floats(r[value_idx] for r in sample)
    ax.bar(labels, y)
    ax.set_ylabel(str(columns[value_idx]))
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", fontsize=6)


def _thin_xticks(ax, x: list, max_ticks: int = 8) -> None:
    n = len(x)
    if n <= max_ticks:
        return
    step = max(1, n // max_ticks)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels([x[i] for i in range(0, n, step)], rotation=45, ha="right", fontsize=7)


def _safe_floats(values) -> list[float]:
    out = []
    for v in values:
        try:
            out.append(float(str(v).replace(",", "")))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _numeric_column_indices(columns: list, rows: list[list]) -> list[int]:
    indices = []
    for col_idx in range(len(columns)):
        sample = [r[col_idx] for r in rows[:_NUMERIC_SAMPLE_SIZE] if col_idx < len(r)]
        if not sample:
            continue
        numeric_count = sum(1 for v in sample if _parses_as_float(v))
        if numeric_count / len(sample) >= _NUMERIC_MIN_RATIO:
            indices.append(col_idx)
    return indices


def _parses_as_float(value) -> bool:
    try:
        float(str(value).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _is_date_column(columns: list, rows: list[list]) -> bool:
    if not columns:
        return False
    if "date" in str(columns[0]).lower():
        return True
    sample = [r[0] for r in rows[:_NUMERIC_SAMPLE_SIZE] if r]
    if not sample:
        return False
    matches = sum(1 for v in sample if _DATE_RE.match(str(v)) or _parses_as_date(v))
    return matches / len(sample) >= _NUMERIC_MIN_RATIO


def _parses_as_date(value) -> bool:
    try:
        datetime.fromisoformat(str(value))
        return True
    except ValueError:
        return False
