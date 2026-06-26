import dataclasses
from pydantic import BaseModel
from tools.base import BaseTool
from services.edgar_service import EdgarFilingService


class TechnicalReportInput(BaseModel):
    ticker: str
    mine_name: str | None = None


class TechnicalReportTool(BaseTool):
    name = "get_mine_technical_report"
    description = (
        "Fetch the SEC S-K 1300 Technical Report Summary (Exhibit 96 of a mining "
        "company's most recent 10-K/20-F) for mineral resource/reserve estimates, mine "
        "life, and project economics. Requires the company's stock ticker; optionally "
        "narrow to one project via mine_name when the filer reports multiple sites."
    )
    input_schema = TechnicalReportInput

    def __init__(self, service: EdgarFilingService | None = None) -> None:
        self._service = service or EdgarFilingService()

    async def run(self, **kwargs) -> dict:
        inp = TechnicalReportInput(**kwargs)
        report = await self._service.get_technical_report_summary(
            ticker=inp.ticker, mine_name=inp.mine_name
        )
        return {"technical_report": dataclasses.asdict(report)}
