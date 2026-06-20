"""Cross-encoder reranker (sentence-transformers).

When reranker_model is empty or reranker_enabled is False, the reranker is a
no-op that returns chunks sorted by their existing ChromaDB similarity score.
"""
import logging

from config import settings
from retrieval.chroma_store import Chunk

logger = logging.getLogger(__name__)


class Reranker:
    """Scores retrieved chunks against a query using a cross-encoder model."""

    def __init__(self) -> None:
        self._enabled = (
            settings.retrieval.reranker_enabled
            and bool(settings.retrieval.reranker_model)
        )
        self._model = None
        if self._enabled:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(settings.retrieval.reranker_model)
            except Exception:
                logger.warning(
                    "Reranker model '%s' could not be loaded; falling back to similarity ranking.",
                    settings.retrieval.reranker_model,
                )
                self._enabled = False

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return chunks sorted by relevance to query (descending score)."""
        if not chunks:
            return chunks
        if not self._enabled or self._model is None:
            return sorted(chunks, key=lambda c: c.score, reverse=True)
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)
        return sorted(chunks, key=lambda c: c.score, reverse=True)
