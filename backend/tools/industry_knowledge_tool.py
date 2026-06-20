from pydantic import BaseModel

from retrieval import Retriever
from tools.base import BaseTool


class IndustryKnowledgeInput(BaseModel):
    query: str
    top_k: int = 5


class IndustryKnowledgeTool(BaseTool):
    """Search the industry knowledge base of mining, construction, and heavy equipment.

    Use during planning to understand what data is needed to answer a question correctly.
    Use during synthesis to interpret what collected signals mean and draw well-grounded conclusions.
    """

    name = "search_industry_knowledge"
    description = (
        "Search the industry knowledge base of mining, construction, and heavy equipment "
        "books and articles. Use during planning to understand what data is needed to answer "
        "a question correctly. Use during synthesis to interpret what collected signals mean "
        "and how to draw well-grounded conclusions."
    )
    input_schema = IndustryKnowledgeInput

    def __init__(self, retriever: Retriever | None = None) -> None:
        self._retriever = retriever or Retriever()

    async def run(self, query: str, top_k: int = 5) -> dict:
        from config import settings

        collection = settings.stores.chroma_knowledge_collection
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
