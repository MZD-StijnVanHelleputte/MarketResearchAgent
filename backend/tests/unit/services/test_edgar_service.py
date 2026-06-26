import pytest
from unittest.mock import AsyncMock, MagicMock
from clients.base_http_client import ClientError
from services.edgar_service import EdgarFilingService, ServiceError

SUBMISSIONS = {
    "name": "FREEPORT-MCMORAN INC",
    "filings": {
        "recent": {
            "form": ["8-K", "10-K", "4", "10-K"],
            "accessionNumber": [
                "0000831259-26-000030",
                "0000831259-26-000012",
                "0001214659-26-007087",
                "0000831259-25-000010",
            ],
            "filingDate": ["2026-06-10", "2026-02-13", "2026-06-02", "2025-02-12"],
        }
    },
}

INDEX_HTML_WITH_EXHIBIT = """
<table>
<tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th><th>Size</th></tr>
<tr><td>38</td><td></td><td><a href="x">a2025trsmorenci-finalxpubl.pdf</a></td><td>EX-96.3</td><td>4436381</td></tr>
<tr><td>39</td><td></td><td><a href="x">a2025trsgrasberg-finalxpubl.pdf</a></td><td>EX-96.4</td><td>3000000</td></tr>
</table>
"""

INDEX_HTML_NO_EXHIBIT = "<table><tr><td>1</td><td></td><td>10k.htm</td><td>10-K</td><td>100</td></tr></table>"


def _mock_edgar(cik="0000831259", submissions=SUBMISSIONS):
    edgar = MagicMock()
    edgar.get_cik_for_ticker = AsyncMock(return_value=cik)
    edgar.get_submissions = AsyncMock(return_value=submissions)
    return edgar


@pytest.mark.asyncio
async def test_happy_path_finds_exhibit_and_returns_excerpt():
    edgar = _mock_edgar()
    edgar.get_document_bytes = AsyncMock(
        side_effect=[
            INDEX_HTML_WITH_EXHIBIT.encode(),
            b"<html><body>Mineral Reserve: 500 Mt copper</body></html>",
        ]
    )
    service = EdgarFilingService(edgar_client=edgar)

    result = await service.get_technical_report_summary("FCX")

    assert result.ticker == "FCX"
    assert result.cik == "0000831259"
    assert result.form_type == "10-K"
    assert result.exhibit_name == "a2025trsmorenci-finalxpubl.pdf"
    assert "Mineral Reserve" in result.excerpt
    assert result.mine_name_matched is None


@pytest.mark.asyncio
async def test_mine_name_disambiguates_among_multiple_exhibits():
    edgar = _mock_edgar()
    edgar.get_document_bytes = AsyncMock(
        side_effect=[
            INDEX_HTML_WITH_EXHIBIT.encode(),
            b"Grasberg copper-gold report text",
        ]
    )
    service = EdgarFilingService(edgar_client=edgar)

    result = await service.get_technical_report_summary("FCX", mine_name="Grasberg")

    assert result.exhibit_name == "a2025trsgrasberg-finalxpubl.pdf"
    assert result.mine_name_matched == "Grasberg"


@pytest.mark.asyncio
async def test_no_cik_raises_service_error():
    edgar = _mock_edgar(cik=None)
    service = EdgarFilingService(edgar_client=edgar)

    with pytest.raises(ServiceError, match="No SEC CIK"):
        await service.get_technical_report_summary("NOTREAL")


@pytest.mark.asyncio
async def test_no_annual_filings_raises_service_error():
    edgar = _mock_edgar(submissions={"name": "X", "filings": {"recent": {
        "form": ["8-K"], "accessionNumber": ["1"], "filingDate": ["2026-01-01"],
    }}})
    service = EdgarFilingService(edgar_client=edgar)

    with pytest.raises(ServiceError, match="No 10-K/20-F filings"):
        await service.get_technical_report_summary("FCX")


@pytest.mark.asyncio
async def test_no_exhibit_96_in_any_scanned_filing_raises_service_error():
    edgar = _mock_edgar()
    edgar.get_document_bytes = AsyncMock(return_value=INDEX_HTML_NO_EXHIBIT.encode())
    service = EdgarFilingService(edgar_client=edgar)

    with pytest.raises(ServiceError, match="No S-K 1300 Technical Report Summary"):
        await service.get_technical_report_summary("FCX")


@pytest.mark.asyncio
async def test_exhibit_fetch_client_error_wrapped_as_service_error():
    edgar = _mock_edgar()
    edgar.get_document_bytes = AsyncMock(
        side_effect=[
            INDEX_HTML_WITH_EXHIBIT.encode(),
            ClientError(500, "boom"),
        ]
    )
    service = EdgarFilingService(edgar_client=edgar)

    with pytest.raises(ServiceError, match="exhibit fetch failed"):
        await service.get_technical_report_summary("FCX")
