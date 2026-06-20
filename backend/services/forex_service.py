import logging
from dataclasses import dataclass, field

from clients.alpha_vantage_client import AlphaVantageClient
from clients.base_http_client import ClientError

logger = logging.getLogger(__name__)

_AV_NOTICE_KEYS = ("Information", "Note", "Error Message")
_INTERVAL_TO_FUNCTION = {
    "daily": "FX_DAILY",
    "weekly": "FX_WEEKLY",
    "monthly": "FX_MONTHLY",
}


class ServiceError(Exception):
    pass


@dataclass
class FxObservation:
    date: str
    open: float | None
    high: float | None
    low: float | None
    close: float | None


@dataclass
class ForexResult:
    from_currency: str
    to_currency: str
    interval: str
    latest: FxObservation | None = None
    rows: list[FxObservation] = field(default_factory=list)
    source: str = "alpha_vantage"


class ForexService:
    """Alpha Vantage FX series service for mining-relevant currency pairs."""

    def __init__(self, client: AlphaVantageClient | None = None) -> None:
        self._client = client or AlphaVantageClient()

    async def get_fx_rates(
        self,
        from_currency: str,
        to_currency: str,
        interval: str = "monthly",
    ) -> ForexResult:
        from_currency = from_currency.strip().upper()
        to_currency = to_currency.strip().upper()
        interval = interval.strip().lower()
        if interval not in _INTERVAL_TO_FUNCTION:
            choices = ", ".join(sorted(_INTERVAL_TO_FUNCTION))
            raise ServiceError(
                f"Unsupported interval '{interval}' for FX rates. Use one of: {choices}."
            )
        function = _INTERVAL_TO_FUNCTION[interval]
        try:
            raw = await self._client.get_fx_series(from_currency, to_currency, function)
        except ClientError as exc:
            raise ServiceError(
                f"Alpha Vantage FX request failed for {from_currency}/{to_currency}: {exc}"
            ) from exc
        return self._normalize(raw, from_currency, to_currency, interval)

    def _normalize(
        self,
        raw: dict,
        from_currency: str,
        to_currency: str,
        interval: str,
    ) -> ForexResult:
        self._raise_on_notice(raw, f"{from_currency}/{to_currency}")
        rows = self._extract_rows(raw)
        latest = next((r for r in rows if r.close is not None), None)
        if latest is None:
            raise ServiceError(
                f"Alpha Vantage returned no usable FX data for {from_currency}/{to_currency}."
            )
        return ForexResult(
            from_currency=from_currency,
            to_currency=to_currency,
            interval=interval,
            latest=latest,
            rows=rows,
        )

    @staticmethod
    def _raise_on_notice(raw: dict, pair: str) -> None:
        if not isinstance(raw, dict):
            raise ServiceError(f"Alpha Vantage returned an invalid FX response for '{pair}'.")
        notices = {k: raw[k] for k in _AV_NOTICE_KEYS if k in raw}
        if notices:
            logger.warning("Alpha Vantage FX notice for %s: %s", pair, notices)
            first = next(iter(notices.values()))
            raise ServiceError(
                f"Alpha Vantage did not return FX data for '{pair}': {first}"
            )

    def _extract_rows(self, raw: dict) -> list[FxObservation]:
        for value in raw.values():
            if isinstance(value, dict) and value:
                first_entry = next(iter(value.values()), None)
                if isinstance(first_entry, dict):
                    return [self._row_from_entry(date, point) for date, point in value.items()]
        return []

    @staticmethod
    def _row_from_entry(date: str, point: dict) -> FxObservation:
        def _get(keys: list[str]) -> float | None:
            for k in keys:
                for field_key, v in point.items():
                    if k in field_key.lower():
                        try:
                            return float(str(v).replace(",", ""))
                        except (TypeError, ValueError):
                            pass
            return None

        return FxObservation(
            date=date,
            open=_get(["open"]),
            high=_get(["high"]),
            low=_get(["low"]),
            close=_get(["close"]),
        )
