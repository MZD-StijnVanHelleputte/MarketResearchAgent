import pytest
from unittest.mock import AsyncMock, MagicMock
from clients.base_http_client import ClientError
from services.competition_service import CompetitionService, CompanyFinancials, Filing, ServiceError

PROFILE = [{"companyName": "Caterpillar Inc", "mktCap": 150e9, "pe": 15.2, "date": "2024-12-31"}]
METRICS = [{"revenueTTM": 67e9, "netIncomeTTM": 6e9, "capexTTM": 2e9}]
EDGAR_HITS = {
    "hits": {
        "hits": [
            {"_source": {"entity_name": "Caterpillar Inc", "form_type": "10-K", "file_date": "2024-02-14", "period_of_report": "2023-12-31"}, "highlight": {}}
        ]
    }
}


def _mock_fmp(profile=PROFILE, metrics=METRICS):
    fmp = MagicMock()
    fmp.get_company_profile = AsyncMock(return_value=profile)
    fmp.get_key_metrics = AsyncMock(return_value=metrics)
    return fmp


def _mock_edgar(hits=EDGAR_HITS):
    edgar = MagicMock()
    edgar.search_filings = AsyncMock(return_value=hits)
    return edgar


@pytest.mark.asyncio
async def test_get_financials_returns_dataclass():
    svc = CompetitionService(fmp_client=_mock_fmp(), edgar_client=_mock_edgar())
    result = await svc.get_financials("CAT")
    assert isinstance(result, CompanyFinancials)
    assert result.ticker == "CAT"
    assert result.name == "Caterpillar Inc"
    assert result.market_cap_usd == 150e9
    assert result.pe_ratio == 15.2


@pytest.mark.asyncio
async def test_get_filings_returns_list():
    svc = CompetitionService(fmp_client=_mock_fmp(), edgar_client=_mock_edgar())
    result = await svc.get_filings("Caterpillar mining")
    assert len(result) == 1
    assert isinstance(result[0], Filing)
    assert result[0].entity_name == "Caterpillar Inc"
    assert result[0].form_type == "10-K"


@pytest.mark.asyncio
async def test_get_financials_raises_service_error_on_client_error():
    fmp = MagicMock()
    fmp.get_company_profile = AsyncMock(side_effect=ClientError(401, "Unauthorized"))
    svc = CompetitionService(fmp_client=fmp, edgar_client=_mock_edgar())
    with pytest.raises(ServiceError):
        await svc.get_financials("CAT")


@pytest.mark.asyncio
async def test_get_financials_handles_empty_metrics():
    svc = CompetitionService(fmp_client=_mock_fmp(metrics=[]), edgar_client=_mock_edgar())
    result = await svc.get_financials("CAT")
    assert result.capex_usd is None
