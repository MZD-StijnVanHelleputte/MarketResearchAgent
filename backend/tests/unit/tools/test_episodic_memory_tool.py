import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from retrieval.chroma_store import Chunk
from retrieval.retriever import StaleChunkWarning
from tools.episodic_memory_tool import EpisodicMemoryTool


def _mock_retriever(chunks=None, warnings=None):
    retriever = MagicMock()
    retriever.retrieve.return_value = (chunks or [], warnings or [])
    return retriever


def _chunk(text="plan text", source="run_123", domain="mining", score=0.85):
    return Chunk(
        text=text, source=source, domain=domain,
        chunk_id="c1", timestamp=datetime.now(timezone.utc).isoformat(), score=score,
    )


@pytest.mark.asyncio
async def test_run_queries_episodic_memory_collection():
    retriever = _mock_retriever(chunks=[_chunk()])
    tool = EpisodicMemoryTool(retriever=retriever)
    await tool.run(query="copper demand research plan")
    call_args = retriever.retrieve.call_args
    assert call_args[0][1] == "episodic_memory"


@pytest.mark.asyncio
async def test_run_passes_top_k():
    retriever = _mock_retriever()
    tool = EpisodicMemoryTool(retriever=retriever)
    await tool.run(query="q", top_k=7)
    assert retriever.retrieve.call_args[1]["top_k"] == 7


@pytest.mark.asyncio
async def test_run_returns_correct_shape():
    c = _chunk(text="past plan content", source="run_99", score=0.9)
    retriever = _mock_retriever(chunks=[c])
    tool = EpisodicMemoryTool(retriever=retriever)
    result = await tool.run(query="copper")

    assert "results" in result
    assert "stale_warnings" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["text"] == "past plan content"
    assert result["results"][0]["source"] == "run_99"
    assert result["results"][0]["score"] == 0.9


@pytest.mark.asyncio
async def test_run_includes_stale_warnings():
    warning = StaleChunkWarning(chunk_id="c1", domain="mining", age_days=400.0, message="old chunk")
    retriever = _mock_retriever(chunks=[], warnings=[warning])
    tool = EpisodicMemoryTool(retriever=retriever)
    result = await tool.run(query="test")

    assert len(result["stale_warnings"]) == 1
    assert result["stale_warnings"][0]["chunk_id"] == "c1"
    assert result["stale_warnings"][0]["age_days"] == 400.0


@pytest.mark.asyncio
async def test_run_empty_collection_returns_empty_results():
    retriever = _mock_retriever(chunks=[], warnings=[])
    tool = EpisodicMemoryTool(retriever=retriever)
    result = await tool.run(query="nothing here")
    assert result == {"results": [], "stale_warnings": []}


def test_tool_metadata():
    tool = EpisodicMemoryTool(retriever=MagicMock())
    assert tool.name == "search_episodic_memory"
    assert "episodic" in tool.description.lower() or "past" in tool.description.lower()
