"""approve_gate must write the post-approval stage immediately, not the
stale pre-gate stage. Otherwise a frontend poll between "approve clicked"
and the background graph resume actually executing sees status="running",
stage="understand" — looking like Gate 1 approval didn't close the
understand phase. See api/routers/gates.py and core/graph.py understand_node.
"""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import BackgroundTasks

from config import settings as cfg
from memory.sqlite_store import SqliteStore
from api.routers.gates import approve_gate


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.stores, "sqlite_path", str(tmp_path / "intel.db"))
    return str(tmp_path / "intel.db")


@pytest.mark.parametrize(
    "gate,expected_stage",
    [(1, "collect"), (2, "synthesize"), (3, "done")],
)
@pytest.mark.asyncio
async def test_approve_gate_writes_next_stage_not_stale_one(tmp_db, gate, expected_stage):
    store = SqliteStore()
    run_id = f"run-gate-{gate}"
    await store.upsert_run(
        run_id, "session-1", "query", status=f"waiting_gate_{gate}", stage="understand",
    )

    background_tasks = BackgroundTasks()
    with patch("api.routers.chat._resume_graph", new=AsyncMock()):
        await approve_gate(run_id, gate, body={}, background_tasks=background_tasks)

    run = await store.get_run(run_id)
    assert run["status"] == "running"
    assert run["stage"] == expected_stage
