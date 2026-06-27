"""Unit tests for the per-tool live Sources-panel writer (event_logger.record_live_source).

The Sources panel must grow source-by-source during collection rather than in
per-domain chunks. record_live_source appends one provisional row to runs.sources
each time a tool returns; graph._persist_partial_sources later swaps a domain's
provisional rows for typed datasets. These tests pin that contract.
"""
import pytest

from config import settings as cfg
from core import event_logger
from core.event_logger import record_live_source, set_run_context
from memory.sqlite_store import SqliteStore


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.stores, "sqlite_path", str(tmp_path / "intel.db"))
    return str(tmp_path / "intel.db")


async def _seed_run(run_id="run1", session_id="sess1", query="copper outlook"):
    await SqliteStore().upsert_run(
        run_id, session_id, query, status="running", stage="collect",
    )


@pytest.mark.asyncio
async def test_records_provisional_source_on_collect(tmp_db):
    await _seed_run()
    set_run_context("run1", stage="collect", domain="competition")

    await record_live_source("get_equity_price", {"ticker": "CAT"}, result={"prices": [1, 2, 3]})

    run = await SqliteStore().get_run("run1")
    assert len(run["sources"]) == 1
    src = run["sources"][0]
    assert src["domain"] == "competition"
    assert src["label"] == "CAT"
    assert src["provisional"] is True
    assert src["failed"] is False
    assert src["count"] == 3


@pytest.mark.asyncio
async def test_failed_tool_records_failed_source(tmp_db):
    await _seed_run()
    set_run_context("run1", stage="collect", domain="competition")

    await record_live_source("search_sec_filings", {"query": "Caterpillar"}, failed=True, reason="HTTP 400")

    run = await SqliteStore().get_run("run1")
    assert run["sources"][0]["failed"] is True
    assert run["sources"][0]["data_type"] == "failed"
    assert run["sources"][0]["reason"] == "HTTP 400"


@pytest.mark.asyncio
async def test_dedupes_identical_provisional_rows(tmp_db):
    await _seed_run()
    set_run_context("run1", stage="collect", domain="competition")

    # A repaired/retried call must not produce a duplicate panel row.
    await record_live_source("get_equity_price", {"ticker": "CAT"}, result={"prices": [1]})
    await record_live_source("get_equity_price", {"ticker": "CAT"}, result={"prices": [1]})

    run = await SqliteStore().get_run("run1")
    assert len(run["sources"]) == 1


@pytest.mark.asyncio
async def test_retry_success_replaces_earlier_failed_row(tmp_db):
    await _seed_run()
    set_run_context("run1", stage="collect", domain="competition")

    # First attempt fails, then a repair retry for the same tool/args succeeds.
    await record_live_source("get_equity_price", {"ticker": "CAT"}, failed=True, reason="HTTP 500")
    await record_live_source("get_equity_price", {"ticker": "CAT"}, result={"prices": [1, 2]})

    run = await SqliteStore().get_run("run1")
    assert len(run["sources"]) == 1
    assert run["sources"][0]["failed"] is False
    assert run["sources"][0]["count"] == 2


@pytest.mark.asyncio
async def test_later_failure_replaces_earlier_success_row(tmp_db):
    await _seed_run()
    set_run_context("run1", stage="collect", domain="competition")

    await record_live_source("get_equity_price", {"ticker": "CAT"}, result={"prices": [1]})
    await record_live_source("get_equity_price", {"ticker": "CAT"}, failed=True, reason="HTTP 500")

    run = await SqliteStore().get_run("run1")
    assert len(run["sources"]) == 1
    assert run["sources"][0]["failed"] is True


@pytest.mark.asyncio
async def test_skips_non_collect_stage(tmp_db):
    await _seed_run()
    # Research/understand tool calls feed planning context, not the Sources panel.
    set_run_context("run1", stage="understand", domain="")

    await record_live_source("web_search", {"query": "largest copper miners"}, result={"results": [1]})

    run = await SqliteStore().get_run("run1")
    assert run["sources"] == []


@pytest.mark.asyncio
async def test_noop_without_run_context(tmp_db):
    # No active run context → must not raise and nothing to persist.
    event_logger._run_ctx.set({})
    await record_live_source("get_equity_price", {"ticker": "CAT"}, result={"prices": [1]})
