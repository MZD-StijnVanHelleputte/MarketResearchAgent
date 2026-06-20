import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone

from retrieval.chroma_store import Chunk
from retrieval.retriever import StaleChunkWarning
from tools.industry_knowledge_tool import IndustryKnowledgeTool


def _mock_retriever(chunks=None, warnings=None):
    retriever = MagicMock()
    retriever.retrieve.return_value = (chunks or [], warnings or [])
    return retriever


def _chunk(text="knowledge text", source="mining_overview.md", domain="mining", score=0.8):
    return Chunk(
        text=text, source=source, domain=domain,
        chunk_id="k1", timestamp=datetime.now(timezone.utc).isoformat(), score=score,
    )


@pytest.mark.asyncio
async def test_run_queries_industry_knowledge_collection():
    retriever = _mock_retriever(chunks=[_chunk()])
    tool = IndustryKnowledgeTool(retriever=retriever)
    await tool.run(query="copper demand cycles")
    call_args = retriever.retrieve.call_args
    assert call_args[0][1] == "industry_knowledge"


@pytest.mark.asyncio
async def test_run_default_top_k_is_5():
    retriever = _mock_retriever()
    tool = IndustryKnowledgeTool(retriever=retriever)
    await tool.run(query="mining")
    assert retriever.retrieve.call_args[1]["top_k"] == 5


@pytest.mark.asyncio
async def test_run_respects_custom_top_k():
    retriever = _mock_retriever()
    tool = IndustryKnowledgeTool(retriever=retriever)
    await tool.run(query="q", top_k=3)
    assert retriever.retrieve.call_args[1]["top_k"] == 3


@pytest.mark.asyncio
async def test_run_returns_correct_shape():
    c = _chunk(text="Caterpillar dominates large mining trucks.", source="competitive_landscape.md", score=0.75)
    retriever = _mock_retriever(chunks=[c])
    tool = IndustryKnowledgeTool(retriever=retriever)
    result = await tool.run(query="competitor mining trucks")

    assert "results" in result
    assert "stale_warnings" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["text"] == "Caterpillar dominates large mining trucks."
    assert result["results"][0]["source"] == "competitive_landscape.md"
    assert result["results"][0]["score"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_run_includes_stale_warnings():
    warning = StaleChunkWarning(chunk_id="k1", domain="mining", age_days=500.0, message="stale doc")
    retriever = _mock_retriever(chunks=[], warnings=[warning])
    tool = IndustryKnowledgeTool(retriever=retriever)
    result = await tool.run(query="test")
    assert len(result["stale_warnings"]) == 1
    assert result["stale_warnings"][0]["chunk_id"] == "k1"


@pytest.mark.asyncio
async def test_run_empty_collection_returns_empty_results():
    retriever = _mock_retriever(chunks=[], warnings=[])
    tool = IndustryKnowledgeTool(retriever=retriever)
    result = await tool.run(query="nothing")
    assert result == {"results": [], "stale_warnings": []}


def test_tool_metadata():
    tool = IndustryKnowledgeTool(retriever=MagicMock())
    assert tool.name == "search_industry_knowledge"
    assert "industry" in tool.description.lower() or "knowledge" in tool.description.lower()
