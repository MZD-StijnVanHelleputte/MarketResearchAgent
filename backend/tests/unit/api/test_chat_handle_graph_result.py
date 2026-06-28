"""Unit tests for _handle_graph_result's empty-content honesty backstop.

A run can reach _handle_graph_result without raising (e.g. a synthesize-phase
redirect/timeout self-loop that exits before producing chapters) yet have
nothing to show. Previously this was persisted as status="done" with no
error, which the frontend trusted and then 404'd fetching a PDF that was
never generated. See core/graph.py synthesize_router / api/routers/chat.py.
"""
from unittest.mock import AsyncMock, MagicMock, patch
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from config import settings as cfg
from memory.sqlite_store import SqliteStore
from api.routers.chat import _handle_graph_result


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.stores, "sqlite_path", str(tmp_path / "intel.db"))
    return str(tmp_path / "intel.db")


def _no_pending_interrupt_snapshot():
    snapshot = MagicMock()
    snapshot.next = ()
    return snapshot


def _gate_interrupt_snapshot(value: dict):
    return SimpleNamespace(
        next=("data_review",),
        tasks=[SimpleNamespace(interrupts=[SimpleNamespace(value=value)])],
    )


@pytest.mark.asyncio
async def test_empty_final_state_persists_error_not_done(tmp_db):
    final = {
        "stage": "synthesize",
        "synthesis_chapters": [],
        "exec_summary": "",
        "warnings": [],
    }
    with patch(
        "core.graph.compiled.aget_state",
        AsyncMock(return_value=_no_pending_interrupt_snapshot()),
    ):
        result = await _handle_graph_result(final, "run1", "sess1", "copper outlook")

    assert result is True
    run = await SqliteStore().get_run("run1")
    assert run["status"] == "error"
    assert run["error"]
    assert "retry" in run["error"].lower()


@pytest.mark.asyncio
async def test_empty_synthesis_recovers_brief_from_chapter_sets(tmp_db):
    """When synthesis was interrupted before committing chapters but chapter_sets
    survives (committed by collect_node), reconstruct a brief instead of erroring."""
    final = {
        "stage": "synthesize",
        "synthesis_chapters": [],
        "exec_summary": "",
        "chapter_sets": {
            "plan_a::commodities": {"domain": "commodities", "text": "Copper demand is rising."},
            "plan_a::competition": {"domain": "competition", "text": "Rivals are expanding."},
        },
        "warnings": [],
    }
    with patch(
        "core.graph.compiled.aget_state",
        AsyncMock(return_value=_no_pending_interrupt_snapshot()),
    ), patch("reports.assembler.Assembler.assemble"), patch(
        "reports.pdf_generator.PdfGenerator.generate"
    ):
        result = await _handle_graph_result(final, "run_recover", "sess1", "copper outlook")

    assert result is True
    run = await SqliteStore().get_run("run_recover")
    assert run["status"] == "done"
    assert not run.get("error")
    assert "Copper demand is rising." in run["brief"]
    assert "Rivals are expanding." in run["brief"]
    assert any("reconstructed" in w.lower() for w in run.get("warnings", []))


@pytest.mark.asyncio
async def test_gate2_interrupt_persists_plans_sources_and_gate_data(tmp_db):
    plan = {
        "plan_id": "plan_gate2",
        "planned_tool_calls": [
            {
                "domain": "commodities",
                "tool": "web_extract",
                "params": {"urls": ["https://example.com/copper"]},
            }
        ],
    }
    final = {
        "stage": "collect",
        "plans": [plan],
        "collection_manifest": {
            "commodities": [
                {
                    "domain": "commodities",
                    "tool": "web_extract",
                    "title": "Copper source",
                    "data_type": "data",
                    "count": 1,
                    "failed": False,
                }
            ]
        },
    }
    gate_data = {
        "gate": 2,
        "domains": [
            {"domain": "commodities", "entities": [], "failed_tools": []}
        ],
    }

    with patch(
        "core.graph.compiled.aget_state",
        AsyncMock(return_value=_gate_interrupt_snapshot(gate_data)),
    ):
        result = await _handle_graph_result(final, "run_gate2", "sess1", "copper outlook")

    assert result is False
    store = SqliteStore()
    await store.log_step_event(
        run_id="run_gate2",
        ts=datetime.now(timezone.utc).isoformat(),
        level="info",
        stage="collect",
        domain="commodities",
        event_type="tool_call",
        label="web_extract [commodities]",
        detail={"call_id": "plan_gate2:commodities:0", "tool": "web_extract"},
    )
    await store.log_step_event(
        run_id="run_gate2",
        ts=datetime.now(timezone.utc).isoformat(),
        level="info",
        stage="collect",
        domain="",
        event_type="progress",
        label="Collecting intelligence from data sources…",
        detail=None,
    )
    run = await store.get_run("run_gate2")
    assert run["status"] == "waiting_gate_2"
    assert run["gate_data"] == gate_data
    assert run["plans"] == [plan]
    assert run["sources"] == final["collection_manifest"]["commodities"]
    assert run["collection_plan"] is not None
    assert run["collection_plan"]["total"] == 1
    assert run["collection_plan"]["succeeded"] == 1
    assert run["collection_plan"]["pending"] == 0


@pytest.mark.asyncio
async def test_real_content_still_persists_done(tmp_db):
    final = {
        "stage": "done",
        "synthesis_chapters": [{"domain": "commodities", "text": "Copper demand is rising."}],
        "exec_summary": "Copper demand is rising into 2026.",
        "warnings": [],
    }
    with patch(
        "core.graph.compiled.aget_state",
        AsyncMock(return_value=_no_pending_interrupt_snapshot()),
    ), patch("reports.assembler.Assembler.assemble"), patch(
        "reports.pdf_generator.PdfGenerator.generate"
    ):
        result = await _handle_graph_result(final, "run2", "sess1", "copper outlook")

    assert result is True
    run = await SqliteStore().get_run("run2")
    assert run["status"] == "done"
    assert not run.get("error")
