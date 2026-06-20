"""Tests for ChromaStore against a real (temp-dir) chromadb instance.

Unlike test_retriever.py, this exercises the actual chromadb engine to verify
add/list/delete behave correctly with fresh client instances per call (the
pattern used in api/routers/knowledge.py, where a new ChromaStore() is
created per request).
"""
from unittest.mock import patch

import pytest

from config import settings
from retrieval.chroma_store import ChromaStore, Document

_COLLECTION = "industry_knowledge"


@pytest.fixture
def chroma_path(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.stores, "chroma_path", str(tmp_path))
    return tmp_path


def _doc(chunk_id: str, source: str) -> Document:
    return Document(text=f"text for {chunk_id}", source=source, domain="mining",
                     chunk_id=chunk_id, timestamp="2026-01-01T00:00:00+00:00")


def test_delete_by_source_persists_across_fresh_client_instances(chroma_path):
    """A delete made by one ChromaStore() instance must be visible to a
    different ChromaStore() instance reading the same on-disk collection —
    this is the exact pattern knowledge.py uses (fresh Retriever() per request)."""
    ChromaStore().add_documents(
        _COLLECTION,
        [_doc("a", "fileA.pdf"), _doc("b", "fileB.pdf")],
        embeddings=[[0.1] * 8, [0.2] * 8],
    )

    deleted = ChromaStore().delete_by_source(_COLLECTION, "fileA.pdf")
    assert deleted == 1

    remaining = ChromaStore().list_sources(_COLLECTION)
    assert [r["source"] for r in remaining] == ["fileB.pdf"]


def test_delete_by_source_returns_zero_when_collection_missing(chroma_path):
    assert ChromaStore().delete_by_source(_COLLECTION, "never-uploaded.pdf") == 0


def test_delete_by_source_propagates_unexpected_errors(chroma_path):
    """Only a missing collection should be swallowed into 0 — any other
    failure (e.g. a malformed where-clause, disk error) must surface instead
    of silently reporting a successful no-op delete."""
    store = ChromaStore()
    store.add_documents(_COLLECTION, [_doc("a", "fileA.pdf")], embeddings=[[0.1] * 8])

    with patch.object(store._client, "get_collection") as mock_get:
        mock_get.return_value.get.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError, match="boom"):
            store.delete_by_source(_COLLECTION, "fileA.pdf")
