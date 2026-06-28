import pytest
from unittest.mock import AsyncMock, MagicMock
from clients.base_http_client import ClientError
from services.macro_service import MacroService, MacroIndicator, MacroObservation, ServiceError

_OBS_RAW = {
    "observations": [
        {"date": "2026-04-01", "value": "5.33"},
        {"date": "2026-03-01", "value": "5.33"},
        {"date": "2026-02-01", "value": "."},  # FRED missing-data sentinel
    ]
}
_INFO_RAW = {
    "seriess": [
        {
            "title": "Federal Funds Effective Rate",
            "units": "Percent",
            "frequency_short": "M",
        }
    ]
}


def _mock_client(obs=None, info=None, raise_exc=None):
    client = MagicMock()
    if raise_exc:
        client.get_observations = AsyncMock(side_effect=raise_exc)
        client.get_series_info = AsyncMock(side_effect=raise_exc)
    else:
        client.get_observations = AsyncMock(return_value=obs or _OBS_RAW)
        client.get_series_info = AsyncMock(return_value=info or _INFO_RAW)
    return client


@pytest.mark.asyncio
async def test_get_indicator_returns_macro_indicator():
    service = MacroService(client=_mock_client())
    result = await service.get_indicator("FEDFUNDS", limit=3)

    assert isinstance(result, MacroIndicator)
    assert result.series_id == "FEDFUNDS"
    assert result.title == "Federal Funds Effective Rate"
    assert result.units == "Percent"
    assert result.frequency == "M"


@pytest.mark.asyncio
async def test_get_indicator_parses_observations():
    service = MacroService(client=_mock_client())
    result = await service.get_indicator("FEDFUNDS", limit=3)

    assert len(result.observations) == 3
    assert isinstance(result.observations[0], MacroObservation)
    assert result.observations[0].date == "2026-04-01"
    assert result.observations[0].value == 5.33


@pytest.mark.asyncio
async def test_dot_value_parsed_as_none():
    service = MacroService(client=_mock_client())
    result = await service.get_indicator("FEDFUNDS", limit=3)

    missing = result.observations[2]
    assert missing.value is None


@pytest.mark.asyncio
async def test_empty_info_falls_back_to_series_id():
    client = _mock_client(info={"seriess": []})
    service = MacroService(client=client)
    result = await service.get_indicator("GDP")

    assert result.title == "GDP"
    assert result.units == ""


@pytest.mark.asyncio
async def test_client_error_wrapped_as_service_error():
    client = _mock_client(raise_exc=ClientError(400, "Bad Request"))
    service = MacroService(client=client)

    with pytest.raises(ServiceError, match="FRED request failed"):
        await service.get_indicator("FEDFUNDS")


@pytest.mark.asyncio
async def test_all_missing_observations_raise_service_error():
    """A window where every value is FRED's '.' sentinel is not a successful fetch."""
    all_missing = {"observations": [
        {"date": "2026-04-01", "value": "."},
        {"date": "2026-03-01", "value": "."},
    ]}
    service = MacroService(client=_mock_client(obs=all_missing))

    with pytest.raises(ServiceError, match="no usable observations"):
        await service.get_indicator("FEDFUNDS")


@pytest.mark.asyncio
async def test_series_id_normalised_to_uppercase():
    """Mixed-case guesses are upcased before the FRED call (FRED maps lowercase to an
    all-missing series instead of erroring)."""
    client = _mock_client()
    service = MacroService(client=client)
    result = await service.get_indicator("fedfunds", limit=3)

    assert result.series_id == "FEDFUNDS"
    client.get_observations.assert_awaited_once_with("FEDFUNDS", 3)
