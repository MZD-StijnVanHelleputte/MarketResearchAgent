"""Chunker: splits text into Document chunks for ChromaDB ingestion.

Three splitting strategies:
  chunk_text()     — sliding-window word chunks (for collected news/filings/web)
  chunk_as_one()   — entire text as a single chunk (for episodic plan summaries)
  chunk_document() — header-aware split then further chunking (for knowledge files)
"""
import re
import uuid
from datetime import datetime, timezone

from retrieval.chroma_store import Document


class Chunker:
    """Word-approximate chunker.  chunk_size and chunk_overlap are in words (~tokens)."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100) -> None:
        self._size = chunk_size
        self._overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_text(self, text: str, source: str, domain: str) -> list[Document]:
        """Sliding-window word chunking.  Used for collected news/web/filing text."""
        words = text.split()
        if not words:
            return []
        chunks: list[Document] = []
        start = 0
        while start < len(words):
            end = min(start + self._size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(self._make(chunk_text, source, domain))
            if end == len(words):
                break
            start = end - self._overlap
        return chunks

    def chunk_as_one(self, text: str, source: str, domain: str) -> list[Document]:
        """Store text as a single chunk.  Used for episodic plan summaries."""
        trimmed = text[:10_000].strip()
        if not trimmed:
            return []
        return [self._make(trimmed, source, domain)]

    def chunk_document(self, text: str, source: str, domain: str) -> list[Document]:
        """Header-aware split for long documents (knowledge base files).

        Splits on Markdown headers (# / ## / ###) first, preserving section
        structure.  Each section is further chunked if it exceeds chunk_size.
        """
        sections = re.split(r"(?m)^(#{1,3} .+)$", text)
        # re.split with a capturing group interleaves headers and bodies
        # result: [pre, header1, body1, header2, body2, ...]
        docs: list[Document] = []
        header = ""
        for part in sections:
            if re.match(r"^#{1,3} ", part):
                header = part.strip()
            else:
                body = part.strip()
                if not body:
                    continue
                section_text = f"{header}\n\n{body}".strip() if header else body
                sub_chunks = self.chunk_text(section_text, source, domain)
                docs.extend(sub_chunks)
        if not docs:
            docs = self.chunk_text(text, source, domain)
        return docs

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make(text: str, source: str, domain: str) -> Document:
        return Document(
            text=text,
            source=source,
            domain=domain,
            chunk_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
