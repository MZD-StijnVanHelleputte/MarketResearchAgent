from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval import Retriever

logger = logging.getLogger(__name__)

# Absolute path to data/ regardless of working directory
_DATA_ROOT = Path(__file__).parent.parent / "data"


@dataclass(frozen=True)
class EntityResolution:
    """The canonical domain an entity belongs to, resolved from master data.

    This is what guarantees one entity → one domain: Caterpillar lives only in
    competitors.json, so it resolves to ``competition`` and can never become a
    customer leaf. See core/domains.py for the domain → master-data mapping.
    """

    label: str        # canonical clean label, e.g. "Caterpillar Inc."
    domain: str       # canonical domain key, e.g. "competition"
    leaf_type: str    # "company" / "commodity" / "distributor"
    key: str          # stable key, e.g. "CAT" or the name
    params: dict = field(default_factory=dict)  # {"ticker": "CAT"} for companies


class MasterDataService:
    """Loads and exposes all version-controlled master data from data/.
    This is the only module that reads files from data/."""

    def __init__(self) -> None:
        from config.settings import settings
        self._equipment = self._load("equipment/komatsu_equipment.json")
        self._operators = self._load("operators/operators.json")
        self._competitors = self._load("competitors/competitors.json")
        self._distributors = self._load("distributors/distributors.json")
        self._construction = self._load("construction/construction.json")
        self._others = self._load("others/others.json")
        self._commodities = self._load_csv(settings.commodity_tickers_path)
        self._entity_index: dict[str, EntityResolution] | None = None

    def _load(self, relative_path: str) -> list[dict]:
        path = _DATA_ROOT / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Master data file not found: {path}. "
                f"Ensure data/{relative_path} exists in the backend directory."
            )
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _load_csv(self, relative_path: str) -> list[dict]:
        path = _DATA_ROOT / relative_path
        if not path.exists():
            raise FileNotFoundError(
                f"Master data file not found: {path}. "
                f"Ensure data/{relative_path} exists in the backend directory."
            )
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]

    def get_equipment(self) -> list[dict]:
        return self._equipment

    def get_operators(self) -> list[dict]:
        return self._operators

    def get_competitors(self) -> list[dict]:
        return self._competitors

    def get_distributors(self) -> list[dict]:
        return self._distributors

    def get_construction(self) -> list[dict]:
        return self._construction

    def get_others(self) -> list[dict]:
        return self._others

    def get_commodities(self) -> list[dict]:
        return self._commodities

    # ------------------------------------------------------------------
    # Entity resolution — the canonical entity → domain index (de-overlap)
    # ------------------------------------------------------------------
    def resolve_entity(self, name_or_ticker: str) -> EntityResolution | None:
        """Return the canonical (label, domain, leaf_type, …) for a known entity,
        or None when it isn't in master data (e.g. a research-surfaced rival).

        Tolerates research-context formatting like ``"Caterpillar Inc. (CAT)"`` by
        also trying the bare name and the parenthesised ticker.
        """
        if not name_or_ticker or not str(name_or_ticker).strip():
            return None
        index = self._entity_resolution_index()
        raw = str(name_or_ticker).strip()
        candidates = [raw]
        m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", raw)
        if m:
            candidates.append(m.group(1).strip())
            candidates.append(m.group(2).strip())
        for cand in candidates:
            hit = index.get(cand.lower())
            if hit is not None:
                return hit
        return None

    def _entity_resolution_index(self) -> dict[str, EntityResolution]:
        if self._entity_index is None:
            self._entity_index = self._build_entity_index()
        return self._entity_index

    def _build_entity_index(self) -> dict[str, EntityResolution]:
        # Iterate domains in ownership-priority order so that, if the same alias
        # ever appears in two master-data files, the higher-priority domain wins
        # ("first one wins" via the `not in index` guard below).
        from core.domains import DOMAINS

        index: dict[str, EntityResolution] = {}
        for spec in sorted(DOMAINS.values(), key=lambda s: s.ownership_priority):
            if not spec.masterdata_source:
                continue
            try:
                records = self.lookup(spec.masterdata_source)
            except ValueError:
                logger.warning(
                    "masterdata index: domain '%s' references unknown source '%s'",
                    spec.key, spec.masterdata_source,
                )
                continue
            for record in records:
                self._index_record(index, record, spec.key, spec.default_leaf_type)
        return index

    @staticmethod
    def _index_record(
        index: dict[str, EntityResolution],
        record: dict,
        domain: str,
        leaf_type: str,
    ) -> None:
        # Company files use lowercase name/ticker; commodity CSV uses Name/Ticker.
        name = str(record.get("name") or record.get("Name") or "").strip()
        ticker = str(record.get("ticker") or record.get("Ticker") or "").strip()
        if not name:
            return
        if leaf_type == "company" and ticker:
            params: dict = {"ticker": ticker}
        elif leaf_type == "commodity" and ticker:
            params = {"symbol": ticker}
        else:
            params = {}
        resolution = EntityResolution(
            label=name,
            domain=domain,
            leaf_type=leaf_type,
            key=(ticker.upper() or name),
            params=params,
        )
        for alias in (name, ticker):
            a = alias.strip().lower()
            if len(a) >= 2 and a not in index:
                index[a] = resolution

    _ENTITY_MAP: dict[str, str] = {
        "distributors": "_distributors",
        "competitors": "_competitors",
        "operators": "_operators",
        "construction": "_construction",
        "others": "_others",
        "equipment": "_equipment",
        "commodities": "_commodities",
    }

    def lookup(
        self,
        entity_type: str,
        region: str = "",
        keyword: str = "",
    ) -> list[dict]:
        """Query an in-memory entity list with optional region and keyword filters.

        Both filters are case-insensitive substring matches across all string values
        (including list elements) in each entity dict.

        Raises ValueError for unknown entity_type.
        """
        if entity_type not in self._ENTITY_MAP:
            raise ValueError(
                f"Unknown entity_type '{entity_type}'. "
                f"Must be one of: {', '.join(self._ENTITY_MAP)}"
            )
        items: list[dict] = getattr(self, self._ENTITY_MAP[entity_type])
        if region:
            items = [item for item in items if self._matches(item, region)]
        if keyword:
            items = [item for item in items if self._matches(item, keyword)]
        return items

    @staticmethod
    def _matches(item: dict, term: str) -> bool:
        term_lower = term.lower()
        for value in item.values():
            if isinstance(value, str) and term_lower in value.lower():
                return True
            if isinstance(value, list) and any(
                isinstance(v, str) and term_lower in v.lower() for v in value
            ):
                return True
        return False

    def load_industry_knowledge(self, retriever: "Retriever") -> int:
        """Seed the industry_knowledge ChromaDB collection from data/knowledge/.

        Skips ingestion if the collection already contains documents (idempotent).
        Returns the number of chunks written (0 if already seeded or no files found).
        """
        from config import settings
        from retrieval.chunker import Chunker

        collection = settings.stores.chroma_knowledge_collection
        if retriever.collection_count(collection) > 0:
            logger.info("Industry knowledge collection already seeded (%d docs). Skipping.", retriever.collection_count(collection))
            return 0

        knowledge_root = _DATA_ROOT / "knowledge"
        if not knowledge_root.exists():
            logger.warning("data/knowledge/ directory not found; skipping knowledge ingestion.")
            return 0

        chunker = Chunker(settings.retrieval.chunk_size, settings.retrieval.chunk_overlap)
        total = 0
        for path in sorted(knowledge_root.rglob("*.md")):
            try:
                text = path.read_text(encoding="utf-8")
                domain = path.parent.name  # e.g. "mining", "construction", "heavy_equipment"
                docs = chunker.chunk_document(text, source=path.name, domain=domain)
                retriever.add(collection, docs)
                total += len(docs)
                logger.info("Ingested %d chunks from %s", len(docs), path.name)
            except Exception as exc:
                logger.warning("Failed to ingest %s: %s", path, exc)

        logger.info("Industry knowledge seeding complete: %d total chunks.", total)
        return total
