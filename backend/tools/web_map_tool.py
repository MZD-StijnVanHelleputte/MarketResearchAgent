import dataclasses
from pydantic import BaseModel, Field
from tools.base import BaseTool
from services.web_search_service import WebSearchService


class WebMapInput(BaseModel):
    url: str = Field(
        description=(
            "Root URL of the site or section to map, "
            "e.g. 'https://www.cat.com/en_US/' or 'https://investor.volvogroup.com/'. "
            "Returns discovered URLs — no page content is extracted."
        )
    )
    instructions: str = Field(
        default="",
        description=(
            "Natural language filter to semantically prioritise relevant URLs. "
            "Example: 'Find pages about mining equipment or autonomous haulage'."
        ),
    )
    limit: int = Field(
        default=50,
        ge=1,
        le=200,
        description="Maximum number of URLs to discover (1–200, default 50).",
    )


class WebMapTool(BaseTool):
    name = "web_map"
    description = (
        "Discover URLs on a website without extracting content via Tavily POST /map. "
        "Use before web_crawl or web_extract to understand site structure and find the "
        "right pages — faster than crawling. Typical workflow: web_map a competitor's IR "
        "section → identify press-release URLs → web_extract those specific pages. "
        "Returns root_url and a list of discovered URLs."
    )
    input_schema = WebMapInput

    def __init__(self, service: WebSearchService | None = None) -> None:
        self._service = service or WebSearchService()

    async def run(self, **kwargs) -> dict:
        inp = WebMapInput(**kwargs)
        result = await self._service.map(
            url=inp.url,
            instructions=inp.instructions,
            limit=inp.limit,
        )
        return dataclasses.asdict(result)
