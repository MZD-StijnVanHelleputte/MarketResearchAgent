import pytest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

from config import settings as cfg
from memory.sqlite_store import SqliteStore
from api.routers.gates import save_to_episodic_memory, _episodic_summary


_PLAN = {
    "plan_id": "plan_001",
    "domain_activations": {"commodities": True, "competition": False},
    "tool_calls": [{"tool": "get_mining_metals_prices"}],
    "rationale": "Focus on copper demand cycle.",
}


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg.stores, "sqlite_path", str(tmp_path / "intel.db"))
    return str(tmp_path / "intel.db")


async def _seed_run(run_id: str, status: str = "done") -> None:
    await SqliteStore().upsert_run(
        run_id, "s1", "copper outlook",
        status=status, stage="done",
        exec_summary="Copper demand remains strong into 2026.",
        brief="Full brief text.",
        plans=[_PLAN],
    )


def test_episodic_summary_includes_query_and_plan():
    run = {"query": "copper outlook", "exec_summary": "Strong demand.", "brief": ""}
    text = _episodic_summary(run, _PLAN)
    assert "copper outlook" in text
    assert "plan_001" in text
    assert "commodities" in text
    assert "get_mining_metals_prices" in text


@pytest.mark.asyncio
async def test_episodic_save_writes_chunk_to_episodic_collection(tmp_db):
    await _seed_run("r1")
    fake_retriever = MagicMock()
    with patch("retrieval.Retriever", return_value=fake_retriever):
        result = await save_to_episodic_memory("r1")

    assert result["status"] == "saved"
    assert result["collection"] == cfg.stores.chroma_episodic_collection
    fake_retriever.add.assert_called_once()
    collection_arg, docs_arg = fake_retriever.add.call_args.args
    assert collection_arg == cfg.stores.chroma_episodic_collection
    assert len(docs_arg) == 1  # chunk_as_one → single chunk


@pytest.mark.asyncio
async def test_episodic_save_rejects_unfinished_run(tmp_db):
    await _seed_run("r2", status="running")
    with pytest.raises(HTTPException) as exc:
        await save_to_episodic_memory("r2")
    assert exc.value.status_code == 409


@pytest.mark.asyncio
async def test_episodic_save_missing_run_404(tmp_db):
    with pytest.raises(HTTPException) as exc:
        await save_to_episodic_memory("does-not-exist")
    assert exc.value.status_code == 404
