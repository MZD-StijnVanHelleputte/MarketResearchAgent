import pytest
import pytest_asyncio

from memory.sqlite_store import SqliteStore


@pytest.fixture
def store(tmp_path):
    return SqliteStore(path=str(tmp_path / "test.db"))


# ── Preferences ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_preference_round_trip(store):
    await store.set_preference("domain_weights", {"competition": 1.5})
    result = await store.get_preference("domain_weights")
    assert result == {"competition": 1.5}


@pytest.mark.asyncio
async def test_preference_overwrite(store):
    await store.set_preference("key", "first")
    await store.set_preference("key", "second")
    assert await store.get_preference("key") == "second"


@pytest.mark.asyncio
async def test_preference_missing_returns_none(store):
    assert await store.get_preference("nonexistent") is None


@pytest.mark.asyncio
async def test_get_all_preferences(store):
    await store.set_preference("a", 1)
    await store.set_preference("b", 2)
    prefs = await store.get_all_preferences()
    assert prefs["a"] == 1
    assert prefs["b"] == 2


# ── Runs ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_round_trip(store):
    await store.upsert_run("r1", "s1", "test query", "running", "understand")
    row = await store.get_run("r1")
    assert row is not None
    assert row["run_id"] == "r1"
    assert row["session_id"] == "s1"
    assert row["status"] == "running"
    assert row["stage"] == "understand"


@pytest.mark.asyncio
async def test_run_upsert_updates_status(store):
    await store.upsert_run("r1", "s1", "q", "running", "understand")
    await store.upsert_run("r1", "s1", "q", "done", "done", brief="The brief.")
    row = await store.get_run("r1")
    assert row["status"] == "done"
    assert row["brief"] == "The brief."


@pytest.mark.asyncio
async def test_run_stores_sources(store):
    sources = [{"domain": "competition", "tool": "NewsAPI", "title": "Test", "url": "http://x"}]
    await store.upsert_run("r1", "s1", "q", "done", "done", sources=sources)
    row = await store.get_run("r1")
    assert row["sources"] == sources


@pytest.mark.asyncio
async def test_get_run_missing_returns_none(store):
    assert await store.get_run("ghost") is None


# ── Sessions ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_sessions_aggregation(store):
    await store.upsert_run("r1", "s1", "q1", "done", "done")
    await store.upsert_run("r2", "s1", "q2", "done", "done")
    await store.upsert_run("r3", "s2", "q3", "done", "done")
    sessions = await store.list_sessions()
    counts = {s["session_id"]: s["run_count"] for s in sessions}
    assert counts["s1"] == 2
    assert counts["s2"] == 1


@pytest.mark.asyncio
async def test_get_session(store):
    await store.upsert_run("r1", "s1", "q", "done", "done")
    await store.upsert_run("r2", "s1", "q", "done", "done")
    session = await store.get_session("s1")
    assert session is not None
    assert session["session_id"] == "s1"
    assert session["run_count"] == 2


@pytest.mark.asyncio
async def test_get_session_missing_returns_none(store):
    assert await store.get_session("ghost") is None


# ── Figures ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_figures_round_trip(store):
    await store.upsert_run("r1", "s1", "q", "running", "collect")
    await store.write_figure("r1", "commodities", "copper_price", 9200.5)
    figures = await store.query_figures("r1")
    assert len(figures) == 1
    assert figures[0]["domain"] == "commodities"
    assert figures[0]["key"] == "copper_price"
    assert figures[0]["value"] == 9200.5


# ── Wipe session ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_wipe_session_removes_figures_only(store):
    await store.upsert_run("r1", "s1", "q", "done", "done")
    await store.write_figure("r1", "comp", "metric", 1.0)
    await store.set_preference("keep_me", True)

    await store.wipe_session("s1")

    assert await store.query_figures("r1") == []
    assert await store.get_run("r1") is not None  # run preserved
    assert await store.get_preference("keep_me") is True  # pref preserved


# ── Cost / call-count fields ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_upsert_and_get_run_cost_and_call_count(store):
    await store.upsert_run(
        "r1", "s1", "query", "done", "done",
        cumulative_cost_usd=0.42,
        api_call_count=17,
    )
    row = await store.get_run("r1")
    assert row["cumulative_cost_usd"] == pytest.approx(0.42)
    assert row["api_call_count"] == 17


@pytest.mark.asyncio
async def test_upsert_run_cost_defaults_to_none(store):
    await store.upsert_run("r1", "s1", "query", "running", "understand")
    row = await store.get_run("r1")
    assert row["cumulative_cost_usd"] is None
    assert row["api_call_count"] is None
