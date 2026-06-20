import logging
from dataclasses import dataclass, field

from clients.alpha_vantage_client import AlphaVantageClient
from clients.base_http_client import ClientError

logger = logging.getLogger(__name__)


class ServiceError(Exception):
    pass


@dataclass
class CommodityObservation:
    date: str
    value: float | None


@dataclass
class CommodityResult:
    symbol: str
    endpoint: str
    interval: str
    unit: str
    latest: CommodityObservation | None = None
    rows: list[CommodityObservation] = field(default_factory=list)
    source: str = "alpha_vantage"


_AV_NOTICE_KEYS = ("Information", "Note", "Error Message")
_METAL_SYMBOLS = {"COPPER", "ALUMINUM", "GOLD", "XAU", "SILVER", "XAG"}
_ENERGY_SYMBOLS = {"WTI", "BRENT", "NATURAL_GAS"}
_BROAD_SYMBOLS = {"ALL_COMMODITIES"}
_AGRICULTURAL_SYMBOLS = {"WHEAT", "CORN"}
_PRECIOUS_METALS = {"GOLD", "XAU", "SILVER", "XAG"}
_SERIES_METALS = {"COPPER", "ALUMINUM"}
_DAILY_WEEKLY_MONTHLY = {"daily", "weekly", "monthly"}
_MONTHLY_QUARTERLY_ANNUAL = {"monthly", "quarterly", "annual"}


class CommodityService:
    """Endpoint-aware Alpha Vantage commodity service.

    Each public method maps to a domain-focused tool and keeps Alpha Vantage as
    the only data source. Fallback market data belongs in separate tools.
    """

    def __init__(self, client: AlphaVantageClient | None = None) -> None:
        self._client = client or AlphaVantageClient()

    async def get_mining_metals_prices(
        self,
        symbol: str,
        interval: str = "monthly",
        include_history: bool = True,
    ) -> CommodityResult:
        symbol = self._validate_symbol(symbol, _METAL_SYMBOLS, "mining metals")
        if symbol in _PRECIOUS_METALS:
            interval = self._validate_interval(interval, _DAILY_WEEKLY_MONTHLY, symbol)
            return await self._get_precious_metal(symbol, interval, include_history)

        interval = self._validate_interval(interval, _MONTHLY_QUARTERLY_ANNUAL, symbol)
        return await self._get_series(symbol, interval)

    async def get_energy_cost_prices(
        self,
        symbol: str,
        interval: str = "monthly",
    ) -> CommodityResult:
        symbol = self._validate_symbol(symbol, _ENERGY_SYMBOLS, "energy costs")
        interval = self._validate_interval(interval, _DAILY_WEEKLY_MONTHLY, symbol)
        return await self._get_series(symbol, interval)

    async def get_broad_commodity_cycle(
        self,
        interval: str = "monthly",
    ) -> CommodityResult:
        interval = self._validate_interval(interval, _MONTHLY_QUARTERLY_ANNUAL, "ALL_COMMODITIES")
        return await self._get_series("ALL_COMMODITIES", interval)

    async def get_agricultural_commodity_prices(
        self,
        symbol: str,
        interval: str = "monthly",
    ) -> CommodityResult:
        symbol = self._validate_symbol(symbol, _AGRICULTURAL_SYMBOLS, "agricultural commodities")
        interval = self._validate_interval(interval, _MONTHLY_QUARTERLY_ANNUAL, symbol)
        return await self._get_series(symbol, interval)

    async def _get_precious_metal(
        self,
        symbol: str,
        interval: str,
        include_history: bool,
    ) -> CommodityResult:
        try:
            if include_history:
                raw = await self._client.get_gold_silver_history(symbol, interval)
                return self._normalize(raw, symbol, "GOLD_SILVER_HISTORY", interval)

            raw = await self._client.get_gold_silver_spot(symbol)
            return self._normalize(raw, symbol, "GOLD_SILVER_SPOT", "spot")
        except ClientError as exc:
            raise ServiceError(f"Alpha Vantage request failed for '{symbol}': {exc}") from exc

    async def _get_series(self, symbol: str, interval: str) -> CommodityResult:
        try:
            raw = await self._client.get_commodity_series(symbol, interval)
        except ClientError as exc:
            raise ServiceError(f"Alpha Vantage request failed for '{symbol}': {exc}") from exc
        return self._normalize(raw, symbol, symbol, interval)

    def _normalize(
        self,
        raw: dict,
        symbol: str,
        endpoint: str,
        interval: str,
    ) -> CommodityResult:
        self._raise_on_notice(raw, symbol)
        rows = self._extract_rows(raw)
        latest = next((row for row in rows if row.value is not None), None)
        unit = self._extract_unit(raw)
        if latest is None:
            raise ServiceError(f"Alpha Vantage returned no usable commodity data for '{symbol}'.")
        return CommodityResult(
            symbol=symbol,
            endpoint=endpoint,
            interval=interval,
            unit=unit,
            latest=latest,
            rows=rows,
        )

    @staticmethod
    def _validate_symbol(symbol: str, allowed: set[str], label: str) -> str:
        normalized = symbol.strip().upper()
        if normalized not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ServiceError(f"Unsupported {label} symbol '{symbol}'. Use one of: {choices}.")
        return normalized

    @staticmethod
    def _validate_interval(interval: str, allowed: set[str], symbol: str) -> str:
        normalized = interval.strip().lower()
        if normalized not in allowed:
            choices = ", ".join(sorted(allowed))
            raise ServiceError(f"Unsupported interval '{interval}' for {symbol}. Use one of: {choices}.")
        return normalized

    @staticmethod
    def _raise_on_notice(raw: dict, symbol: str) -> None:
        if not isinstance(raw, dict):
            raise ServiceError(f"Alpha Vantage returned an invalid response for '{symbol}'.")
        notices = {k: raw[k] for k in _AV_NOTICE_KEYS if k in raw}
        if notices:
            logger.warning("Alpha Vantage commodity notice for %s: %s", symbol, notices)
            first = next(iter(notices.values()))
            raise ServiceError(f"Alpha Vantage did not return commodity data for '{symbol}': {first}")

    @staticmethod
    def _extract_unit(raw: dict) -> str:
        for key, value in raw.items():
            if key.lower() == "unit" and value is not None:
                return str(value)
        for meta_key in ("Meta Data", "metadata", "meta_data"):
            meta = raw.get(meta_key)
            if isinstance(meta, dict):
                for key, value in meta.items():
                    if "unit" in key.lower() and value is not None:
                        return str(value)
        return ""

    def _extract_rows(self, raw: dict) -> list[CommodityObservation]:
        data = raw.get("data")
        if isinstance(data, list):
            return [self._row_from_mapping(point) for point in data if isinstance(point, dict)]

        for value in raw.values():
            if isinstance(value, list):
                return [self._row_from_mapping(point) for point in value if isinstance(point, dict)]
            if isinstance(value, dict):
                nested = self._rows_from_timeseries(value)
                if nested:
                    return nested
                single = self._spot_from_mapping(value)
                if single:
                    return [single]

        single = self._spot_from_mapping(raw)
        return [single] if single else []

    def _rows_from_timeseries(self, series: dict) -> list[CommodityObservation]:
        rows: list[CommodityObservation] = []
        for date_value, point in series.items():
            if not isinstance(point, dict):
                continue
            value = self._first_numeric_value(point)
            rows.append(CommodityObservation(date=str(date_value), value=value))
        return rows

    def _row_from_mapping(self, point: dict) -> CommodityObservation:
        date_value = (
            point.get("date")
            or point.get("timestamp")
            or point.get("time")
            or point.get("datetime")
            or ""
        )
        return CommodityObservation(date=str(date_value), value=self._first_numeric_value(point))

    def _spot_from_mapping(self, raw: dict) -> CommodityObservation | None:
        value = self._first_numeric_value(raw)
        if value is None:
            return None
        date_value = raw.get("date") or raw.get("timestamp") or raw.get("time") or ""
        return CommodityObservation(date=str(date_value), value=value)

    @staticmethod
    def _first_numeric_value(point: dict) -> float | None:
        preferred_fragments = ("value", "price", "close", "rate")
        for fragment in preferred_fragments:
            for key, value in point.items():
                if fragment in key.lower():
                    parsed = CommodityService._parse_float(value)
                    if parsed is not None:
                        return parsed
        for value in point.values():
            parsed = CommodityService._parse_float(value)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_float(value) -> float | None:
        if value in (None, "", "."):
            return None
        try:
            return float(str(value).replace(",", ""))
        except (TypeError, ValueError):
            return None
