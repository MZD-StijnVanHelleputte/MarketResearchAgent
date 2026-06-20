import pytest
from unittest.mock import MagicMock, patch

from api.routers.knowledge import _process_job
from services.knowledge_jobs import knowledge_job_store


@pytest.mark.asyncio
async def test_process_job_reaches_done_for_good_file():
    job = knowledge_job_store.create("notes.md", "mining")

    with patch("api.routers.knowledge.convert_to_markdown", return_value="# Title\n\nbody text"), \
         patch("api.routers.knowledge.Retriever") as mock_retriever_cls:
        mock_retriever_cls.return_value.add = MagicMock()
        await _process_job(job.job_id, "notes.md", b"irrelevant", "mining", job.created_at)

    result = knowledge_job_store.get(job.job_id)
    assert result["stage"] == "done"
    assert result["chunks_added"] == 1
    assert result["error"] is None


@pytest.mark.asyncio
async def test_process_job_reaches_error_for_bad_conversion():
    job = knowledge_job_store.create("bad.xlsx", "mining")

    with patch(
        "api.routers.knowledge.convert_to_markdown",
        side_effect=ValueError("Unsupported file type '.xlsx'"),
    ):
        await _process_job(job.job_id, "bad.xlsx", b"irrelevant", "mining", job.created_at)

    result = knowledge_job_store.get(job.job_id)
    assert result["stage"] == "error"
    assert "Unsupported file type" in result["error"]


@pytest.mark.asyncio
async def test_process_job_reaches_error_when_chunking_produces_nothing():
    job = knowledge_job_store.create("empty.md", "mining")

    with patch("api.routers.knowledge.convert_to_markdown", return_value="   "):
        await _process_job(job.job_id, "empty.md", b"irrelevant", "mining", job.created_at)

    result = knowledge_job_store.get(job.job_id)
    assert result["stage"] == "error"
    assert "no chunks" in result["error"]


@pytest.mark.asyncio
async def test_process_job_skips_store_if_source_deleted_before_chunking():
    """A delete recorded after the job was created but before the embed/store
    step must cancel the job instead of letting it re-insert chunks."""
    job = knowledge_job_store.create("stale.md", "mining")
    knowledge_job_store.record_delete("stale.md")

    with patch("api.routers.knowledge.convert_to_markdown", return_value="# Title\n\nbody text"), \
         patch("api.routers.knowledge.Retriever") as mock_retriever_cls:
        await _process_job(job.job_id, "stale.md", b"irrelevant", "mining", job.created_at)

    result = knowledge_job_store.get(job.job_id)
    assert result["stage"] == "cancelled"
    mock_retriever_cls.return_value.add.assert_not_called()


@pytest.mark.asyncio
async def test_process_job_skips_store_if_source_deleted_while_waiting_for_embedding_lock():
    """A delete that arrives after chunking (but before the embedding lock is
    acquired) must still cancel the job, not just one recorded before chunking."""
    job = knowledge_job_store.create("late.md", "mining")

    def convert_then_delete(filename, raw):
        # Simulate the delete request landing while this job is mid-pipeline.
        knowledge_job_store.record_delete("late.md")
        return "# Title\n\nbody text"

    with patch("api.routers.knowledge.convert_to_markdown", side_effect=convert_then_delete), \
         patch("api.routers.knowledge.Retriever") as mock_retriever_cls:
        await _process_job(job.job_id, "late.md", b"irrelevant", "mining", job.created_at)

    result = knowledge_job_store.get(job.job_id)
    assert result["stage"] == "cancelled"
    mock_retriever_cls.return_value.add.assert_not_called()
