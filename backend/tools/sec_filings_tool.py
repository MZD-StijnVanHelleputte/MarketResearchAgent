import dataclasses
from pydantic import BaseModel
from tools.base import BaseTool
from services.competition_service import CompetitionService


class SecFilingsInput(BaseModel):
    query: str
    forms: str = "10-K,10-Q,8-K"


class SecFilingsTool(BaseTool):
    name = "search_sec_filings"
    description = (
        "Search SEC EDGAR filings by keyword. Returns matching filing metadata "
        "including entity name, form type, filing date, and reporting period."
    )
    input_schema = SecFilingsInput

    def __init__(self, service: CompetitionService | None = None) -> None:
        self._service = service or CompetitionService()

    async def run(self, **kwargs) -> dict:
        inp = SecFilingsInput(**kwargs)
        filings = await self._service.get_filings(query=inp.query, forms=inp.forms)
        return {"filings": [dataclasses.asdict(f) for f in filings]}
