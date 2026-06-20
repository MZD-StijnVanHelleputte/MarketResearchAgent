"""Retriever: the single retrieval interface used by tools, agents, and core/.

Combines ChromaStore, Embedder, Reranker, and Chunker.
Applies the staleness guard and named-entity confidence discount.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from config import settings
from retrieval.chroma_store import Chunk, ChromaStore, Document
from retrieval.chunker import Chunker
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker


@dataclass
class StaleChunkWarning:
    chunk_id: str
    domain: str
    age_days: float
    message: str


class Retriever:
    """Unified retrieval interface for all three ChromaDB collections.

    Usage:
        r = Retriever()
        chunks, warnings = r.retrieve("copper demand cycles", "industry_knowledge")
        r.add("collected_run123", docs)
        r.delete_collection("collected_run123")
    """

    def __init__(self) -> None:
        self._store = ChromaStore()
        self._embedder = Embedder()
        self._reranker = Reranker()
        self._chunker = Chunker(
            settings.retrieval.chunk_size,
            settings.retrieval.chunk_overlap,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        collection: str,
        top_k: int | None = None,
    ) -> tuple[list[Chunk], list[StaleChunkWarning]]:
        """Retrieve the most relevant chunks for *query* from *collection*.

        Returns (chunks, warnings).  Chunks are reranked and capped at top_k.
        Warnings are emitted for chunks that exceed the per-domain staleness window.
        The named_entity_confidence_discount is applied to industry_knowledge chunks.
        """
        if top_k is None:
            top_k = settings.retrieval.top_k

        query_embedding = self._embedder.embed(query)
        # Fetch extra candidates so the reranker has room to work
        candidates = self._store.query(collection, query_embedding, top_k * 2)
        reranked = self._reranker.rerank(query, candidates)[:top_k]

        knowledge_collection = settings.stores.chroma_knowledge_collection
        discount = settings.retrieval.named_entity_confidence_discount

        chunks: list[Chunk] = []
        warnings: list[StaleChunkWarning] = []

        for chunk in reranked:
            age_days = self._chunk_age_days(chunk)
            window = settings.retrieval.staleness_window_days.get(chunk.domain, 365)
            if age_days > window:
                msg = (
                    f"Chunk {chunk.chunk_id} from '{chunk.source}' "
                    f"(domain={chunk.domain}) is {age_days:.0f} days old "
                    f"(limit={window} days)."
                )
                warnings.append(
                    StaleChunkWarning(chunk.chunk_id, chunk.domain, age_days, msg)
                )
            if collection == knowledge_collection:
                chunk.score *= discount
            chunks.append(chunk)

        return chunks, warnings

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(
        self,
        collection: str,
        docs: list[Document],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> None:
        """Embed *docs* and write them to *collection*."""
        if not docs:
            return
        texts = [d.text for d in docs]
        embeddings = self._embedder.embed_many(texts, on_progress=on_progress)
        self._store.add_documents(collection, docs, embeddings)

    # ------------------------------------------------------------------
    # Admin
    # ------------------------------------------------------------------

    def delete_collection(self, collection: str) -> None:
        self._store.delete_collection(collection)

    def collection_count(self, collection: str) -> int:
        return self._store.collection_count(collection)

    def list_sources(self, collection: str) -> list[dict]:
        return self._store.list_sources(collection)

    def delete_by_source(self, collection: str, source: str) -> int:
        return self._store.delete_by_source(collection, source)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_age_days(chunk: Chunk) -> float:
        try:
            ts = datetime.fromisoformat(chunk.timestamp)
            now = datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            return (now - ts).total_seconds() / 86_400
        except Exception:
            return 0.0
