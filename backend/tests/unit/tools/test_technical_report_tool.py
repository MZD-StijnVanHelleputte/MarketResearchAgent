import pytest
from unittest.mock import AsyncMock, MagicMock
from services.edgar_service import TechnicalReportSummary
from tools.technical_report_tool import TechnicalReportTool

REPORT = TechnicalReportSummary(
    ticker="FCX", cik="0000831259", company_name="FREEPORT-MCMORAN INC",
    form_type="10-K", filing_date="2026-02-13",
    exhibit_name="a2025trsmorenci-finalxpubl.pdf",
    exhibit_url="https://www.sec.gov/Archives/edgar/data/831259/x/a2025trsmorenci-finalxpubl.pdf",
    excerpt="Mineral Reserve: 500 Mt copper", mine_name_matched=None,
)


def _mock_service(result=REPORT):
    svc = MagicMock()
    svc.get_technical_report_summary = AsyncMock(return_value=result)
    return svc


@pytest.mark.asyncio
async def test_run_returns_wrapped_dict():
    tool = TechnicalReportTool(service=_mock_service())
    result = await tool.run(ticker="FCX")
    assert result["technical_report"]["ticker"] == "FCX"
    assert result["technical_report"]["exhibit_name"] == "a2025trsmorenci-finalxpubl.pdf"


@pytest.mark.asyncio
async def test_run_passes_mine_name_through():
    svc = _mock_service()
    tool = TechnicalReportTool(service=svc)
    await tool.run(ticker="FCX", mine_name="Morenci")
    svc.get_technical_report_summary.assert_called_once_with(ticker="FCX", mine_name="Morenci")


@pytest.mark.asyncio
async def test_run_defaults_mine_name_to_none():
    svc = _mock_service()
    tool = TechnicalReportTool(service=svc)
    await tool.run(ticker="FCX")
    svc.get_technical_report_summary.assert_called_once_with(ticker="FCX", mine_name=None)


def test_tool_metadata():
    tool = TechnicalReportTool()
    assert tool.name == "get_mine_technical_report"
    assert "exhibit 96" in tool.description.lower() or "technical report" in tool.description.lower()
