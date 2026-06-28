import pytest
from unittest.mock import AsyncMock, MagicMock
from clients.base_http_client import ClientError
from services import competition_service
from services.competition_service import CompetitionService, CompanyFinancials, Filing, ServiceError

# FMP /stable/profile and /stable/key-metrics-ttm response shapes.
PROFILE = [{"companyName": "Caterpillar Inc", "marketCap": 150e9, "price": 380.0,
            "currency": "USD", "industry": "Machinery"}]
METRICS = [{"revenueTTM": 67e9, "netIncomeTTM": 6e9, "capexTTM": 2e9, "peRatioTTM": 15.2}]
# yfinance get_company_overview shape.
OVERVIEW = {"symbol": "CAT", "name": "Caterpillar Inc", "price": 381.0, "market_cap": 151e9,
            "revenue": 68e9, "net_income": 6.1e9, "capex": -2.1e9, "pe_ratio": 15.5,
            "currency": "USD", "industry": "Machinery", "date": "2026-06-27"}
# EDGAR full-text-search hit shape.
EDGAR_HITS = {
    "hits": {
        "hits": [
            {"_source": {"display_names": ["CATERPILLAR INC (CAT) (CIK 0000018230)"],
                         "form": "10-K", "file_date": "2024-02-14",
                         "period_ending": "2023-12-31", "file_type": "10-K"},
             "_id": "0000018230-24-000012", "highlight": {}}
        ]
    }
}


def _mock_fmp(profile=PROFILE, metrics=METRICS):
    fmp = MagicMock()
    fmp.get_company_profile = AsyncMock(return_value=profile)
    fmp.get_key_metrics = AsyncMock(return_value=metrics)
    return fmp


def _mock_yf(overview=OVERVIEW):
    yf = MagicMock()
    yf.get_company_overview = AsyncMock(return_value=overview)
    return yf


def _mock_edgar(hits=EDGAR_HITS):
    edgar = MagicMock()
    edgar.search_filings = AsyncMock(return_value=hits)
    return edgar


@pytest.fixture
def free_tier(monkeypatch):
    monkeypatch.setattr(competition_service.settings, "fmp_tier", "free")


@pytest.fixture
def premium_tier(monkeypatch):
    monkeypatch.setattr(competition_service.settings, "fmp_tier", "premium")


@pytest.mark.asyncio
async def test_free_tier_uses_yfinance(free_tier):
    fmp = _mock_fmp()
    svc = CompetitionService(fmp_client=fmp, edgar_client=_mock_edgar(), yf_client=_mock_yf())
    result = await svc.get_financials("CAT")
    assert isinstance(result, CompanyFinancials)
    assert result.name == "Caterpillar Inc"
    assert result.market_cap_usd == 151e9
    assert result.revenue_usd == 68e9
    assert result.pe_ratio == 15.5
    fmp.get_company_profile.assert_not_called()  # FMP not hit on free tier


@pytest.mark.asyncio
async def test_free_tier_falls_back_to_fmp_when_yfinance_fails(free_tier):
    yf = MagicMock()
    yf.get_company_overview = AsyncMock(side_effect=RuntimeError("yfinance down"))
    svc = CompetitionService(fmp_client=_mock_fmp(), edgar_client=_mock_edgar(), yf_client=yf)
    result = await svc.get_financials("CAT")
    assert result.market_cap_usd == 150e9  # came from FMP
    assert result.pe_ratio == 15.2


@pytest.mark.asyncio
async def test_premium_tier_uses_fmp(premium_tier):
    yf = _mock_yf()
    svc = CompetitionService(fmp_client=_mock_fmp(), edgar_client=_mock_edgar(), yf_client=yf)
    result = await svc.get_financials("CAT")
    assert result.market_cap_usd == 150e9  # FMP value
    assert result.pe_ratio == 15.2
    yf.get_company_overview.assert_not_called()


@pytest.mark.asyncio
async def test_premium_tier_falls_back_to_yfinance(premium_tier):
    fmp = MagicMock()
    fmp.get_company_profile = AsyncMock(side_effect=ClientError(429, "Too Many Requests"))
    svc = CompetitionService(fmp_client=fmp, edgar_client=_mock_edgar(), yf_client=_mock_yf())
    result = await svc.get_financials("CAT")
    assert result.market_cap_usd == 151e9  # came from yfinance fallback
    assert result.revenue_usd == 68e9


@pytest.mark.asyncio
async def test_raises_service_error_when_both_sources_fail(free_tier):
    yf = MagicMock()
    yf.get_company_overview = AsyncMock(side_effect=RuntimeError("yfinance down"))
    fmp = MagicMock()
    fmp.get_company_profile = AsyncMock(side_effect=ClientError(401, "Unauthorized"))
    svc = CompetitionService(fmp_client=fmp, edgar_client=_mock_edgar(), yf_client=yf)
    with pytest.raises(ServiceError):
        await svc.get_financials("CAT")


@pytest.mark.asyncio
async def test_premium_tier_handles_empty_metrics(premium_tier):
    svc = CompetitionService(fmp_client=_mock_fmp(metrics=[]), edgar_client=_mock_edgar(),
                             yf_client=_mock_yf())
    result = await svc.get_financials("CAT")
    assert result.capex_usd is None


@pytest.mark.asyncio
async def test_get_filings_returns_list():
    svc = CompetitionService(fmp_client=_mock_fmp(), edgar_client=_mock_edgar(), yf_client=_mock_yf())
    result = await svc.get_filings("Caterpillar mining")
    assert len(result) == 1
    assert isinstance(result[0], Filing)
    assert result[0].entity_name == "CATERPILLAR INC (CAT)"
    assert result[0].form_type == "10-K"
