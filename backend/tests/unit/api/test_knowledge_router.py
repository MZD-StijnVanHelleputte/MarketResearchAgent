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
        await _process_job(job.job_id, "notes.md", b"irrelevant", "mining")

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
        await _process_job(job.job_id, "bad.xlsx", b"irrelevant", "mining")

    result = knowledge_job_store.get(job.job_id)
    assert result["stage"] == "error"
    assert "Unsupported file type" in result["error"]


@pytest.mark.asyncio
async def test_process_job_reaches_error_when_chunking_produces_nothing():
    job = knowledge_job_store.create("empty.md", "mining")

    with patch("api.routers.knowledge.convert_to_markdown", return_value="   "):
        await _process_job(job.job_id, "empty.md", b"irrelevant", "mining")

    result = knowledge_job_store.get(job.job_id)
    assert result["stage"] == "error"
    assert "no chunks" in result["error"]
