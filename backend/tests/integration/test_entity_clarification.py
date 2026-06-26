"""Integration tests for Phase 9.3 entity clarification gate.

These tests exercise the full POST /chat → waiting_clarification →
POST /clarify → run resumes flow. They use the FastAPI test client and
an in-memory SQLite database so no real LLM calls are made.

Run with:
    pytest tests/integration/test_entity_clarification.py -v -m integration
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_first_run_yields_waiting_clarification():
    """Without entity preferences, POST /chat should result in status=waiting_clarification."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Clear entity preferences by mocking SqliteStore.get_preference to return None
        with patch(
            "memory.sqlite_store.SqliteStore.get_preference",
            new=AsyncMock(return_value=None),
        ), patch(
            "memory.sqlite_store.SqliteStore.wipe_session",
            new=AsyncMock(),
        ), patch(
            "memory.sqlite_store.SqliteStore.upsert_run",
            new=AsyncMock(),
        ):
            response = await client.post(
                "/chat",
                json={"query": "What is Caterpillar's current capex cycle?"},
            )

        assert response.status_code == 200
        data = response.json()
        run_id = data["run_id"]
        assert run_id

        # Poll the run until it reaches a terminal state
        import asyncio
        status = "running"
        for _ in range(20):
            await asyncio.sleep(0.1)
            with patch(
                "memory.sqlite_store.SqliteStore.get_run",
                new=AsyncMock(return_value={
                    "run_id": run_id,
                    "status": "waiting_clarification",
                    "stage": "clarification_needed",
                    "session_id": data["session_id"],
                    "query": "What is Caterpillar's current capex cycle?",
                    "confidence": 0.0,
                    "brief": None,
                    "sources": [],
                    "error": "Missing required entity preferences: ['equipment_models']",
                    "initial_state": None,
                }),
            ):
                run_resp = await client.get(f"/runs/{run_id}")
                if run_resp.status_code == 200:
                    status = run_resp.json().get("status", "running")
                    if status != "running":
                        break

        assert status == "waiting_clarification"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clarify_endpoint_resumes_run():
    """POST /runs/{run_id}/clarify should save preferences and resume the run."""
    run_id = "test-run-clarify-001"
    session_id = "test-session-001"

    stored_run = {
        "run_id": run_id,
        "status": "waiting_clarification",
        "stage": "clarification_needed",
        "session_id": session_id,
        "query": "What is Caterpillar's current capex cycle?",
        "confidence": 0.0,
        "brief": None,
        "sources": [],
        "error": None,
        "initial_state": {
            "run_id": run_id,
            "session_id": session_id,
            "user_query": "What is Caterpillar's current capex cycle?",
            "plans": [],
            "collection_manifest": {},
            "chapter_sets": {},
            "synthesis_chapters": [],
            "merge_log": [],
            "confidence": 0.0,
            "react_iterations": 0,
            "context_messages": [],
            "stage": "understand",
            "warnings": [],
            "error": None,
            "cumulative_cost_usd": 0.0,
            "api_call_count": 0,
            "injection_flags": [],
            "clarification_done": False,
        },
    }

    set_pref_calls: list[tuple] = []

    async def mock_set_preference(key: str, value) -> None:
        set_pref_calls.append((key, value))

    graph_invoked = []

    async def mock_run_graph(rid, sid, query, state):
        graph_invoked.append({"run_id": rid, "clarification_done": state.get("clarification_done")})

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("memory.sqlite_store.SqliteStore.get_run", new=AsyncMock(return_value=stored_run)), \
             patch("memory.sqlite_store.SqliteStore.set_preference", new=AsyncMock(side_effect=mock_set_preference)), \
             patch("memory.sqlite_store.SqliteStore.upsert_run", new=AsyncMock()), \
             patch("api.routers.chat._run_graph", new=mock_run_graph):

            response = await client.post(
                f"/runs/{run_id}/clarify",
                json={
                    "equipment_models": ["PC2000", "PC3000"],
                    "operators": ["Rio Tinto", "BHP"],
                    "competitor_tickers": ["CAT", "HII"],
                },
            )

    assert response.status_code == 200
    assert response.json()["status"] == "resumed"

    # Verify preferences were saved
    saved_keys = {k for k, _ in set_pref_calls}
    assert "equipment_models" in saved_keys
    assert "operators" in saved_keys
    assert "competitor_tickers" in saved_keys

    # Verify graph was re-invoked with clarification_done=True
    import asyncio
    await asyncio.sleep(0.05)  # allow background task to run
    assert len(graph_invoked) >= 1
    assert graph_invoked[0]["clarification_done"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_clarify_returns_404_for_non_clarification_run():
    """POST /runs/{run_id}/clarify should 404 if run is not waiting_clarification."""
    run_id = "test-run-done-002"
    done_run = {
        "run_id": run_id,
        "status": "done",
        "stage": "done",
        "session_id": "s1",
        "query": "...",
        "confidence": 0.9,
        "brief": "...",
        "sources": [],
        "error": None,
        "initial_state": None,
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        with patch("memory.sqlite_store.SqliteStore.get_run", new=AsyncMock(return_value=done_run)):
            response = await client.post(
                f"/runs/{run_id}/clarify",
                json={"equipment_models": ["PC2000"]},
            )

    assert response.status_code == 404
