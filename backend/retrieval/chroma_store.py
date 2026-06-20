"""ChromaDB wrapper + shared Document and Chunk dataclasses.

This is the only file in the codebase that imports chromadb directly.
All three collections (collected_*, episodic_memory, industry_knowledge) go through here.
Embeddings are precomputed by the Embedder and passed in — ChromaStore does not embed.
"""
from dataclasses import dataclass, field
from pathlib import Path

import chromadb

from config import settings


@dataclass
class Document:
    """Write unit: a single chunk to be stored in ChromaDB."""
    text: str
    source: str
    domain: str
    chunk_id: str
    timestamp: str   # ISO 8601 UTC


@dataclass
class Chunk:
    """Read unit: a retrieved chunk with its similarity score."""
    text: str
    source: str
    domain: str
    chunk_id: str
    timestamp: str
    score: float = 0.0


class ChromaStore:
    """Thin wrapper over chromadb.PersistentClient.

    All three RAG collections are accessed through this class:
      - collected_{run_id}  : per-run unstructured data (wiped per chat)
      - episodic_memory     : persistent past reports + plans
      - industry_knowledge  : persistent industry books/articles
    """

    def __init__(self) -> None:
        chroma_path = Path(settings.stores.chroma_path)
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_path))

    def add_documents(
        self,
        collection: str,
        docs: list[Document],
        embeddings: list[list[float]],
    ) -> None:
        if not docs:
            return
        col = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )
        col.add(
            ids=[d.chunk_id for d in docs],
            documents=[d.text for d in docs],
            embeddings=embeddings,
            metadatas=[
                {"source": d.source, "domain": d.domain, "timestamp": d.timestamp}
                for d in docs
            ],
        )

    def query(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[Chunk]:
        try:
            col = self._client.get_collection(name=collection)
        except Exception:
            return []

        n = min(top_k, col.count())
        if n == 0:
            return []

        results = col.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[Chunk] = []
        for i, text in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            # chromadb cosine distance → similarity score (1 - distance)
            score = max(0.0, 1.0 - distance)
            chunks.append(
                Chunk(
                    text=text,
                    source=meta.get("source", ""),
                    domain=meta.get("domain", ""),
                    chunk_id=results["ids"][0][i],
                    timestamp=meta.get("timestamp", ""),
                    score=score,
                )
            )
        return chunks

    def delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(name=collection)
        except Exception:
            pass

    def collection_count(self, collection: str) -> int:
        try:
            col = self._client.get_collection(name=collection)
            return col.count()
        except Exception:
            return 0

    def list_sources(self, collection: str) -> list[dict]:
        """Return one entry per unique source with chunk count, domain, and earliest timestamp."""
        try:
            col = self._client.get_collection(name=collection)
            result = col.get(include=["metadatas"])
        except Exception:
            return []
        sources: dict[str, dict] = {}
        for meta in result["metadatas"]:
            src = meta.get("source", "unknown")
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "domain": meta.get("domain", ""),
                    "chunk_count": 0,
                    "added_at": meta.get("timestamp", ""),
                }
            sources[src]["chunk_count"] += 1
        return list(sources.values())

    def delete_by_source(self, collection: str, source: str) -> int:
        """Delete all chunks whose metadata.source == source. Returns deleted count."""
        try:
            col = self._client.get_collection(name=collection)
            result = col.get(where={"source": source}, include=[])
            ids = result["ids"]
            if ids:
                col.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0
