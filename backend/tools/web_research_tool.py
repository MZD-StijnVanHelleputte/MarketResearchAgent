import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.web_search_service import WebSearchService


class WebResearchInput(BaseModel):
    query: str = Field(
        description=(
            "Research question or topic for deep multi-source analysis. "
            "Examples: 'competitive landscape of autonomous mining trucks 2025', "
            "'lithium demand outlook and impact on mining capex in Chile', "
            "'Caterpillar vs Komatsu market share in large mining equipment'."
        )
    )
    model: str = Field(
        default="auto",
        pattern=r"^(mini|pro|auto)$",
        description=(
            "'auto' (default) — API selects based on complexity. "
            "'mini' (~30 s) — single-topic, targeted. "
            "'pro' (~60–120 s) — comprehensive multi-angle analysis and comparisons."
        ),
    )


class WebResearchTool(BaseTool):
    name = "web_research"
    description = (
        "AI-synthesized multi-source research with citations via Tavily POST /research. "
        "Returns a structured report grounded in web sources — NOT just snippets. "
        "Use for deep strategic topics: competitive landscape analysis, market outlooks, "
        "commodity demand forecasts, geopolitical risk assessments. "
        "Takes 30–120 seconds (budget time accordingly). "
        "Use model='pro' for complex comparisons; model='mini' for focused single-topic research. "
        "Returns query, model, report text, and list of citation URLs."
    )
    input_schema = WebResearchInput

    def __init__(self, service: WebSearchService | None = None) -> None:
        self._service = service or WebSearchService()

    async def run(self, **kwargs) -> dict:
        inp = WebResearchInput(**kwargs)
        result = await self._service.research(query=inp.query, model=inp.model)
        return dataclasses.asdict(result)
