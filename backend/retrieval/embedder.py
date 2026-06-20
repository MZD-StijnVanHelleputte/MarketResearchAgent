"""Embedder: wraps LLMClient.embed() for use within the retrieval pipeline.

Uses Mistral's mistral-embed model via the existing LLMClient abstraction.
No sentence-transformers needed for embedding — the Mistral API handles it.
"""
from typing import Callable

from models.llm_client import LLMClient


class Embedder:
    """Converts text to embedding vectors via the configured LLM provider."""

    def __init__(self) -> None:
        self._client = LLMClient()

    def embed(self, text: str) -> list[float]:
        return self._client.embed(text)

    def embed_many(
        self,
        texts: list[str],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        return self._client.embed_batch(texts, on_progress=on_progress)
