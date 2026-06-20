"""Public API for the retrieval layer.

Only import from this module; do not reach into sub-modules directly.
"""
from retrieval.chroma_store import Chunk, Document
from retrieval.retriever import Retriever, StaleChunkWarning

__all__ = ["Retriever", "Document", "Chunk", "StaleChunkWarning"]
