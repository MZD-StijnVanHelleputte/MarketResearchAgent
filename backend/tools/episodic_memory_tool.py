from pydantic import BaseModel

from retrieval import Retriever
from tools.base import BaseTool


class EpisodicMemoryInput(BaseModel):
    query: str
    top_k: int = 3


class EpisodicMemoryTool(BaseTool):
    """Search past successful research reports and their execution plans.

    Use during planning to find examples of plans that worked for similar questions.
    Use during synthesis to see how past reports handled similar signals and conclusions.
    """

    name = "search_episodic_memory"
    description = (
        "Search past successful research reports and their execution plans. "
        "Use during planning to find examples of plans that worked for similar questions. "
        "Use during synthesis to see how past reports handled similar signals and conclusions."
    )
    input_schema = EpisodicMemoryInput

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever or Retriever()

    async def run(self, query: str, top_k: int = 3) -> dict:
        from config import settings

        collection = settings.stores.chroma_episodic_collection
        chunks, warnings = self._retriever.retrieve(query, collection, top_k=top_k)
        return {
            "results": [
                {"text": c.text, "source": c.source, "domain": c.domain, "score": round(c.score, 4)}
                for c in chunks
            ],
            "stale_warnings": [
                {"chunk_id": w.chunk_id, "age_days": round(w.age_days, 1), "message": w.message}
                for w in warnings
            ],
        }
