"""Golden-file and unit tests for the Retriever.

All ChromaStore and Embedder calls are mocked — no real ChromaDB or Mistral needed.
"""
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from retrieval.chroma_store import Chunk, Document
from retrieval.retriever import Retriever, StaleChunkWarning


def _make_chunk(chunk_id="c1", domain="mining", score=0.9, days_old=0) -> Chunk:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_old)).isoformat()
    return Chunk(text="some text", source="test", domain=domain, chunk_id=chunk_id, timestamp=ts, score=score)


def _make_retriever(chunks: list[Chunk], embed_return=None):
    """Return a Retriever with mocked ChromaStore, Embedder, and Reranker."""
    embed_return = embed_return or [0.1] * 1024

    with patch("retrieval.retriever.ChromaStore") as MockStore, \
         patch("retrieval.retriever.Embedder") as MockEmbedder, \
         patch("retrieval.retriever.Reranker") as MockReranker, \
         patch("retrieval.retriever.Chunker"):
        store = MagicMock()
        store.query.return_value = chunks
        store.collection_count.return_value = len(chunks)
        MockStore.return_value = store

        embedder = MagicMock()
        embedder.embed.return_value = embed_return
        embedder.embed_many.return_value = [embed_return]
        MockEmbedder.return_value = embedder

        reranker = MagicMock()
        reranker.rerank.side_effect = lambda query, c: sorted(c, key=lambda x: x.score, reverse=True)
        MockReranker.return_value = reranker

        retriever = Retriever()
        retriever._store = store
        retriever._embedder = embedder
        retriever._reranker = reranker

    return retriever, store, embedder


def test_retrieve_returns_top_k_chunks():
    chunks = [_make_chunk(f"c{i}", score=float(i) / 10) for i in range(10)]
    retriever, store, _ = _make_retriever(chunks)
    store.query.return_value = chunks

    result_chunks, warnings = retriever.retrieve("test query", "industry_knowledge", top_k=3)
    assert len(result_chunks) == 3
    assert not warnings


def test_retrieve_passes_correct_collection_to_store():
    retriever, store, embedder = _make_retriever([])
    retriever.retrieve("query", "episodic_memory", top_k=2)
    store.query.assert_called_once()
    call_args = store.query.call_args
    assert call_args[0][0] == "episodic_memory"


def test_retrieve_applies_named_entity_discount_for_knowledge_collection():
    chunk = _make_chunk("c1", domain="mining", score=1.0)
    retriever, store, _ = _make_retriever([chunk])
    store.query.return_value = [chunk]

    with patch("retrieval.retriever.settings") as mock_settings:
        mock_settings.retrieval.top_k = 5
        mock_settings.retrieval.staleness_window_days = {}
        mock_settings.retrieval.named_entity_confidence_discount = 0.7
        mock_settings.stores.chroma_knowledge_collection = "industry_knowledge"

        result_chunks, _ = retriever.retrieve("query", "industry_knowledge", top_k=1)

    assert len(result_chunks) == 1
    assert result_chunks[0].score == pytest.approx(0.7)


def test_retrieve_emits_staleness_warning_for_old_chunk():
    old_chunk = _make_chunk("old", domain="mining", score=0.9, days_old=400)
    retriever, store, _ = _make_retriever([old_chunk])
    store.query.return_value = [old_chunk]

    with patch("retrieval.retriever.settings") as mock_settings:
        mock_settings.retrieval.top_k = 5
        mock_settings.retrieval.staleness_window_days = {"mining": 30}
        mock_settings.retrieval.named_entity_confidence_discount = 1.0
        mock_settings.stores.chroma_knowledge_collection = "industry_knowledge"

        _, warnings = retriever.retrieve("query", "collected_run1", top_k=1)

    assert len(warnings) == 1
    assert isinstance(warnings[0], StaleChunkWarning)
    assert warnings[0].chunk_id == "old"
    assert warnings[0].age_days > 30


def test_retrieve_no_staleness_warning_for_fresh_chunk():
    fresh_chunk = _make_chunk("fresh", domain="mining", score=0.9, days_old=1)
    retriever, store, _ = _make_retriever([fresh_chunk])
    store.query.return_value = [fresh_chunk]

    with patch("retrieval.retriever.settings") as mock_settings:
        mock_settings.retrieval.top_k = 5
        mock_settings.retrieval.staleness_window_days = {"mining": 30}
        mock_settings.retrieval.named_entity_confidence_discount = 1.0
        mock_settings.stores.chroma_knowledge_collection = "industry_knowledge"

        _, warnings = retriever.retrieve("query", "collected_run1", top_k=1)

    assert warnings == []


def test_add_calls_embed_many_and_store():
    retriever, store, embedder = _make_retriever([])
    doc = Document(text="hello", source="s", domain="d", chunk_id="x", timestamp="2026-01-01T00:00:00+00:00")
    retriever.add("my_collection", [doc])
    embedder.embed_many.assert_called_once_with(["hello"], on_progress=None)
    store.add_documents.assert_called_once()


def test_add_empty_list_does_nothing():
    retriever, store, embedder = _make_retriever([])
    retriever.add("col", [])
    embedder.embed_many.assert_not_called()
    store.add_documents.assert_not_called()


def test_delete_collection_delegates_to_store():
    retriever, store, _ = _make_retriever([])
    retriever.delete_collection("collected_run1")
    store.delete_collection.assert_called_once_with("collected_run1")


def test_collection_count_delegates_to_store():
    retriever, store, _ = _make_retriever([])
    store.collection_count.return_value = 42
    assert retriever.collection_count("episodic_memory") == 42
