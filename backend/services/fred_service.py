import asyncio
import dataclasses
from dataclasses import dataclass

from clients.base_http_client import ClientError
from clients.fred_client import FredClient, get_fred_client


class ServiceError(Exception):
    pass


@dataclass
class FredSeriesMeta:
    series_id: str
    title: str
    units: str
    frequency: str
    popularity: int
    observation_start: str
    observation_end: str


@dataclass
class FredObservation:
    date: str
    value: float | None


@dataclass
class FredObservationsResult:
    series_id: str
    title: str
    units: str
    frequency: str
    observation_start: str
    observation_end: str
    observations: list[FredObservation]


@dataclass
class FredRelease:
    release_id: int
    name: str
    press_release: bool
    link: str


@dataclass
class FredCategory:
    category_id: int
    name: str
    parent_id: int | None


def _parse_series_meta(s: dict) -> FredSeriesMeta:
    return FredSeriesMeta(
        series_id=s.get("id", ""),
        title=s.get("title", ""),
        units=s.get("units", ""),
        frequency=s.get("frequency_short", ""),
        popularity=int(s.get("popularity", 0)),
        observation_start=s.get("observation_start", ""),
        observation_end=s.get("observation_end", ""),
    )


def _parse_release(r: dict) -> FredRelease:
    return FredRelease(
        release_id=int(r.get("id", 0)),
        name=r.get("name", ""),
        press_release=bool(r.get("press_release", False)),
        link=r.get("link", ""),
    )


def _parse_category(c: dict) -> FredCategory:
    parent = c.get("parent_id")
    return FredCategory(
        category_id=int(c.get("id", 0)),
        name=c.get("name", ""),
        parent_id=int(parent) if parent not in (None, 0, "") else None,
    )


class FredService:
    """Business logic wrapper for the expanded FRED API endpoints."""

    def __init__(self, client: FredClient | None = None) -> None:
        self._client = client or get_fred_client()

    async def search_series(self, search_text: str, limit: int = 20) -> list[FredSeriesMeta]:
        try:
            raw = await self._client.search_series(search_text, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FRED series search failed: {exc}") from exc
        return [_parse_series_meta(s) for s in raw.get("seriess", [])]

    async def get_observations(
        self,
        series_id: str,
        observation_start: str | None = None,
        observation_end: str | None = None,
        limit: int = 100,
        units: str = "lin",
        frequency: str | None = None,
    ) -> FredObservationsResult:
        # FRED series_ids are canonically uppercase; a mixed-case guess is silently mapped
        # to an all-missing series instead of erroring, so normalise before the call.
        series_id = series_id.strip().upper()
        try:
            obs_raw, info_raw = await asyncio.gather(
                self._client.get_observations_extended(
                    series_id,
                    observation_start=observation_start,
                    observation_end=observation_end,
                    limit=limit,
                    units=units,
                    frequency=frequency,
                ),
                self._client.get_series_info(series_id),
            )
        except ClientError as exc:
            raise ServiceError(f"FRED observations failed for '{series_id}': {exc}") from exc

        series_list = info_raw.get("seriess", [{}])
        meta = series_list[0] if series_list else {}

        observations = [
            FredObservation(
                date=o["date"],
                value=float(o["value"]) if o.get("value") not in (".", None, "") else None,
            )
            for o in obs_raw.get("observations", [])
        ]

        # All-missing windows (every value ".") are not a successful fetch — fail so the
        # agent treats it as a miss rather than reporting an empty series as real data.
        if not any(o.value is not None for o in observations):
            raise ServiceError(f"FRED series '{series_id}' returned no usable observations.")

        return FredObservationsResult(
            series_id=series_id,
            title=meta.get("title", series_id),
            units=meta.get("units", ""),
            frequency=meta.get("frequency_short", ""),
            observation_start=obs_raw.get("observation_start", ""),
            observation_end=obs_raw.get("observation_end", ""),
            observations=observations,
        )

    async def list_releases(self, limit: int = 50) -> list[FredRelease]:
        try:
            raw = await self._client.get_releases(limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FRED releases list failed: {exc}") from exc
        return [_parse_release(r) for r in raw.get("releases", [])]

    async def get_release_with_series(
        self, release_id: int, limit: int = 20
    ) -> dict:
        try:
            release_raw, series_raw = await asyncio.gather(
                self._client.get_release(release_id),
                self._client.get_release_series(release_id, limit=limit),
            )
        except ClientError as exc:
            raise ServiceError(f"FRED release {release_id} failed: {exc}") from exc

        releases = release_raw.get("releases", [{}])
        release = _parse_release(releases[0]) if releases else FredRelease(release_id, "", False, "")
        series = [_parse_series_meta(s) for s in series_raw.get("seriess", [])]

        return {
            "release": dataclasses.asdict(release),
            "series": [dataclasses.asdict(s) for s in series],
        }

    async def browse_category(
        self, category_id: int = 0, include_series: bool = False, series_limit: int = 20
    ) -> dict:
        try:
            if include_series:
                cat_raw, children_raw, series_raw = await asyncio.gather(
                    self._client.get_category(category_id),
                    self._client.get_category_children(category_id),
                    self._client.get_category_series(category_id, limit=series_limit),
                )
            else:
                cat_raw, children_raw = await asyncio.gather(
                    self._client.get_category(category_id),
                    self._client.get_category_children(category_id),
                )
                series_raw = {"seriess": []}
        except ClientError as exc:
            raise ServiceError(f"FRED category {category_id} failed: {exc}") from exc

        categories = cat_raw.get("categories", [{}])
        category = _parse_category(categories[0]) if categories else FredCategory(category_id, "", None)
        children = [_parse_category(c) for c in children_raw.get("categories", [])]
        series = [_parse_series_meta(s) for s in series_raw.get("seriess", [])]

        return {
            "category": dataclasses.asdict(category),
            "children": [dataclasses.asdict(c) for c in children],
            "series": [dataclasses.asdict(s) for s in series],
        }

    async def get_series_by_tags(self, tag_names: str, limit: int = 20) -> list[FredSeriesMeta]:
        try:
            raw = await self._client.get_tags_series(tag_names, limit=limit)
        except ClientError as exc:
            raise ServiceError(f"FRED tags series failed for '{tag_names}': {exc}") from exc
        return [_parse_series_meta(s) for s in raw.get("seriess", [])]

    async def get_series_updates(
        self, limit: int = 20, filter_value: str = "macro"
    ) -> list[FredSeriesMeta]:
        try:
            raw = await self._client.get_series_updates(limit=limit, filter_value=filter_value)
        except ClientError as exc:
            raise ServiceError(f"FRED series updates failed: {exc}") from exc
        return [_parse_series_meta(s) for s in raw.get("seriess", [])]
