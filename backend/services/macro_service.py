import asyncio
from dataclasses import dataclass

from clients.fred_client import FredClient, get_fred_client
from clients.base_http_client import ClientError


class ServiceError(Exception):
    pass


@dataclass
class MacroObservation:
    date: str
    value: float | None  # None when FRED returns "." (data not available)


@dataclass
class MacroIndicator:
    series_id: str
    title: str
    units: str
    frequency: str
    observations: list[MacroObservation]


class MacroService:
    """Business logic wrapper over FredClient."""

    def __init__(self, client: FredClient | None = None) -> None:
        self._client = client or get_fred_client()

    async def get_indicator(self, series_id: str, limit: int = 10) -> MacroIndicator:
        # FRED series_ids are canonically uppercase; the planner often guesses mixed case,
        # which FRED silently maps to an all-missing series rather than erroring.
        series_id = series_id.strip().upper()
        try:
            obs_raw, info_raw = await asyncio.gather(
                self._client.get_observations(series_id, limit),
                self._client.get_series_info(series_id),
            )
        except ClientError as exc:
            raise ServiceError(f"FRED request failed for '{series_id}': {exc}") from exc

        series_list = info_raw.get("seriess", [{}])
        meta = series_list[0] if series_list else {}

        observations = [
            MacroObservation(
                date=o["date"],
                value=float(o["value"]) if o.get("value") not in (".", None, "") else None,
            )
            for o in obs_raw.get("observations", [])
        ]

        # A series with no usable values (FRED returns "." for every observation in the
        # window) is not a successful fetch — surface it as an error so the agent treats
        # it as a miss instead of reporting an empty indicator as real data.
        if not any(o.value is not None for o in observations):
            raise ServiceError(f"FRED series '{series_id}' returned no usable observations.")

        return MacroIndicator(
            series_id=series_id,
            title=meta.get("title", series_id),
            units=meta.get("units", ""),
            frequency=meta.get("frequency_short", ""),
            observations=observations,
        )
