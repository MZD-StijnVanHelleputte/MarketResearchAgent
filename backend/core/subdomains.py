"""Subdomain decomposition for the 3-tier hierarchical synthesis.

Given a merged per-domain chapter, enumerate the entities/subdomains it covers
(each competitor, each commodity, each distributor, …) and assemble the
entity-specific evidence used to write a Tier-1 leaf analysis.

Design notes:
  - Collection is unchanged: all evidence already lives in the per-run Chroma
    collection (`collected_{run_id}`) plus the MergedChapter's datasets/figures.
    This module only *re-slices* that evidence per entity at synthesis time.
  - Entities come from the plan's entity_manifest (explicitly researched) and the
    version-controlled master data, filtered to those actually present in the
    collected evidence so we never emit empty sub-sections.
  - general_search has no fixed master-data entity list, so its "entities" are
    themes extracted from this run's actual evidence via one LLM call (see
    `_theme_candidates`). Domains that fail to yield >=2 entities/themes (or any
    other unlisted domain) return [] so the caller falls back to the legacy
    single-chapter synthesis path.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from config import settings
from core.domains import masterdata_source, ownership_order
from core.schemas import MergedChapter
from models.llm_client import LLMClient
from models.usage import llm_usage
from prompts.synthesize_prompt import THEME_EXTRACTION_TEMPLATE

logger = logging.getLogger(__name__)

# Domains with no fixed master-data entity list, decomposed into LLM-derived
# themes instead of named entities (see _theme_candidates).
_THEMATIC_DOMAINS = {"general_search"}


@dataclass
class Subdomain:
    """One entity within a domain, plus everything needed to find its evidence."""

    key: str                       # stable key, e.g. "CAT", "GC", "Sandvik"
    label: str                     # human label, e.g. "Caterpillar Inc."
    aliases: set[str] = field(default_factory=set)  # strings that identify the entity
    query_hint: str = ""           # retrieval query seed
    segment: str = ""              # customer segment, e.g. "Mining", "Construction", "Others"

    def __post_init__(self) -> None:
        self.aliases = {a for a in self.aliases if a and len(a) >= 2}
        if not self.query_hint:
            self.query_hint = self.label


@dataclass
class EntityEvidence:
    """Evidence slice handed to the Tier-1 synthesis for one subdomain."""

    retrieved_chunks: list = field(default_factory=list)   # clean Chunk objects
    datasets: list[dict] = field(default_factory=list)
    figures: dict[str, str] = field(default_factory=dict)
    citations: list[dict] = field(default_factory=list)    # structured {id, title, url, publisher}
    injection_flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------

def enumerate_subdomains(
    domain: str,
    merged_chapter: MergedChapter,
    plan: dict,
    masterdata,
) -> tuple[list[Subdomain], dict]:
    """Return the entities to decompose *domain* into, plus any LLM usage incurred.

    Returns ([], {}) for non-decomposable domains and for decomposable domains
    where fewer than 2 entities/themes have supporting evidence — in both cases
    the caller keeps today's single-chapter behavior.
    """
    if domain not in settings.synthesis.decomposable_domains:
        return [], {}

    evidence_text = _evidence_text(merged_chapter)
    usage: dict = {}

    if domain in _THEMATIC_DOMAINS:
        manifest = plan.get("entity_manifest", {}) if isinstance(plan, dict) else {}
        demand_side = [str(c).strip() for c in (manifest.get("demand_side_companies") or []) if str(c).strip()]
        try:
            candidates, usage = _theme_candidates(domain, merged_chapter, demand_side)
        except Exception as exc:  # never let enumeration break synthesis
            logger.warning("enumerate_subdomains(%s): theme extraction failed: %s", domain, exc)
            return [], {}
    else:
        manifest = plan.get("entity_manifest", {}) if isinstance(plan, dict) else {}
        builder = _CANDIDATE_BUILDERS.get(domain)
        if builder is None:
            return [], {}
        try:
            candidates = builder(manifest, masterdata)
        except Exception as exc:  # never let enumeration break synthesis
            logger.warning("enumerate_subdomains(%s): candidate build failed: %s", domain, exc)
            return [], {}

    # Keep entities that are actually present in the collected evidence so we
    # don't emit empty sub-sections (a planned ticker with no returned data).
    present = [c for c in candidates if _mentions(evidence_text, c.aliases)]

    # De-duplicate by key, preserving order.
    seen: set[str] = set()
    unique: list[Subdomain] = []
    for c in present:
        if c.key not in seen:
            seen.add(c.key)
            unique.append(c)

    cap = settings.synthesis.max_subdomains_per_domain
    if len(unique) > cap:
        logger.info(
            "enumerate_subdomains(%s): capping %d entities to %d",
            domain, len(unique), cap,
        )
        unique = unique[:cap]

    # A single entity adds no hierarchy over the rollup — treat as degenerate.
    return (unique, usage) if len(unique) >= 2 else ([], usage)


# ---------------------------------------------------------------------------
# Evidence assembly
# ---------------------------------------------------------------------------

def assemble_entity_evidence(
    sub: Subdomain,
    merged_chapter: MergedChapter,
    retriever,
    collection: str,
    guardrails,
    query: str,
    top_k: int | None = None,
) -> EntityEvidence:
    """Slice the domain's evidence down to one entity for its Tier-1 analysis."""
    evidence = EntityEvidence()

    # 1. Entity-focused retrieval from the per-run collected store.
    retrieval_query = f"{sub.query_hint}: {query}"
    try:
        raw_chunks, _stale = retriever.retrieve(
            retrieval_query,
            collection,
            top_k=top_k if top_k is not None else settings.retrieval.top_k,
        )
    except Exception as exc:
        logger.debug("assemble_entity_evidence: retrieval failed for %s: %s", sub.key, exc)
        raw_chunks = []

    # 2. Filter injection-tainted chunks (same guard as the domain path).
    for chunk in raw_chunks:
        chunk_text = getattr(chunk, "text", None) or (
            chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        )
        warning = guardrails.scan_for_injection(chunk_text)
        if warning:
            evidence.injection_flags.append(
                f"Chunk filtered ({merged_chapter.domain}/{sub.key}): {warning}"
            )
        else:
            evidence.retrieved_chunks.append(chunk)

    # 3. Datasets / figures / citations attributable to this entity.
    evidence.datasets = [
        ds for ds in merged_chapter.datasets if _dataset_mentions(ds, sub.aliases)
    ]
    evidence.figures = {
        k: v for k, v in merged_chapter.figures.items()
        if _mentions(f"{k} {v}".lower(), sub.aliases)
    }
    evidence.citations = [
        c for c in merged_chapter.citations
        if _mentions(f"{c.get('title', '')} {c.get('publisher', '')}".lower(), sub.aliases)
    ]
    return evidence


def classify_entity_domain(
    dataset: dict,
    masterdata,
    demand_side_companies: list[str] | None = None,
    manifest: dict | None = None,
) -> tuple[str | None, str]:
    """Return (true_domain, segment) for *dataset* by checking it against every
    master-data-backed domain in ownership-priority order; the first alias hit wins.

    Used to catch datasets that were collected by one domain's agent (e.g. a broad
    web search) but are actually about an entity that belongs to a different domain
    (e.g. a competitor surfaced while collecting mining-operator data). Returns
    (None, "") when nothing matches, so callers keep the dataset's original domain.
    The segment element is retained for call-site compatibility and is always "".
    """
    manifest = manifest or {}

    # Try the canonical master-data resolution first (same mechanism plan_merger
    # uses at plan-build time), using the ticker/symbol embedded in the dataset's
    # series_id ("tool:TICKER[:period]") or label. Without this, a company that
    # resolve_entity correctly maps to e.g. "mining_operators" could still get
    # reclassified into "competition" by the candidate-alias scan below, since
    # that scan trusts free-text manifest lists and competition has a higher
    # ownership priority than most customer domains.
    if masterdata is not None:
        for ident in _dataset_identifiers(dataset):
            res = masterdata.resolve_entity(ident)
            if res is not None:
                return res.domain, ""

    for domain in ownership_order():
        if not masterdata_source(domain):
            continue  # research-derived domains aren't reclassification targets
        builder = _CANDIDATE_BUILDERS.get(domain)
        if builder is None:
            continue
        candidates = builder(manifest, masterdata)
        if any(_dataset_mentions(dataset, c.aliases) for c in candidates):
            return domain, ""

    if demand_side_companies:
        aliases = {str(c).strip() for c in demand_side_companies if str(c).strip()}
        if _dataset_mentions(dataset, aliases):
            return "general_search", ""

    return None, ""


def group_datasets_by_entity(
    domain: str,
    datasets: list[dict],
    plan: dict,
    masterdata,
) -> list[dict]:
    """Bucket a domain's Gate-2 datasets under the named entity each one mentions.

    Reuses the same candidate builders and alias matching as synthesis-time
    decomposition, so Gate 2's "Competition → Caterpillar / John Deere" grouping
    is consistent with the eventual chapter structure. Each dataset is assigned to
    the first candidate whose aliases it mentions; unattributable datasets fall into
    a trailing "General" bucket.

    Returns [{"label": str, "datasets": [...]}] with only non-empty buckets.
    Thematic / non-decomposable domains (no fixed entity list, and no LLM call at a
    gate) return a single {"label": "General", "datasets": datasets} bucket.
    """
    if not datasets:
        return []

    builder = _CANDIDATE_BUILDERS.get(domain)
    if builder is None:
        return [{"label": "General", "datasets": datasets}]

    manifest = plan.get("entity_manifest", {}) if isinstance(plan, dict) else {}
    try:
        candidates = builder(manifest, masterdata)
    except Exception as exc:  # never let grouping break the gate
        logger.warning("group_datasets_by_entity(%s): candidate build failed: %s", domain, exc)
        return [{"label": "General", "datasets": datasets}]

    buckets: dict[str, dict] = {}
    order: list[str] = []
    general: list[dict] = []
    for ds in datasets:
        matched = next(
            (c for c in candidates if _dataset_mentions(ds, c.aliases)), None
        )
        if matched is None:
            general.append(ds)
            continue
        # Segment-tagged candidates (e.g. customers: Mining/Construction/Others) get a
        # prefixed bucket label so Gate 2 and the PDF group companies by segment.
        bucket_label = f"{matched.segment}: {matched.label}" if matched.segment else matched.label
        if bucket_label not in buckets:
            buckets[bucket_label] = {"label": bucket_label, "datasets": []}
            order.append(bucket_label)
        buckets[bucket_label]["datasets"].append(ds)

    result = [buckets[label] for label in order]
    if general:
        result.append({"label": "General", "datasets": general})
    return result


# ---------------------------------------------------------------------------
# Per-domain candidate builders
# ---------------------------------------------------------------------------

def _competition_candidates(manifest: dict, masterdata) -> list[Subdomain]:
    competitors = masterdata.get_competitors()
    by_ticker = {
        str(c.get("ticker", "")).upper(): c for c in competitors if c.get("ticker")
    }
    cands: list[Subdomain] = []

    # Master-data competitors are the canonical, clean-label set.
    for c in competitors:
        name = c.get("name", "")
        ticker = str(c.get("ticker", "") or "")
        if not name:
            continue
        cands.append(Subdomain(
            key=ticker.upper() or name,
            label=name,
            aliases={name, ticker},
            query_hint=f"{name} ({ticker}) competitor strategy financials",
        ))

    # Researched competitors not in master data (e.g. a newly surfaced rival) —
    # use the researched name as the label so we don't fall back to a bare ticker.
    # Guard against the manifest's free-text "competitors"/"tickers" lists pulling
    # in entities that are actually canonically owned by another domain (e.g. a
    # mining operator like BHP, which resolve_entity correctly maps to
    # "mining_operators" via data/operators/operators.json) — without this check,
    # competition's higher ownership priority would steal their datasets at
    # classify_entity_domain() time.
    seen_names = {c.get("name", "") for c in competitors}
    for c in manifest.get("competitors", []) or []:
        label = str(c).strip()
        if not label or label in seen_names:
            continue
        res = masterdata.resolve_entity(label) if masterdata is not None else None
        if res is not None and res.domain != "competition":
            continue
        cands.append(Subdomain(
            key=label, label=label, aliases={label},
            query_hint=f"{label} competitor strategy",
        ))

    # Any planned ticker not in master data and not already covered above by name.
    for t in manifest.get("tickers", []) or []:
        tu = str(t).strip().upper()
        if not tu or tu in by_ticker:
            continue
        res = masterdata.resolve_entity(tu) if masterdata is not None else None
        if res is not None and res.domain != "competition":
            continue
        cands.append(Subdomain(
            key=tu, label=tu, aliases={tu},
            query_hint=f"{tu} competitor",
        ))
    return cands


def _commodities_candidates(manifest: dict, masterdata) -> list[Subdomain]:
    commodities = masterdata.get_commodities()
    cands: list[Subdomain] = []
    for c in commodities:
        name = c.get("Name", "")
        ticker = str(c.get("Ticker", "") or "")
        if not name:
            continue
        cands.append(Subdomain(
            key=ticker or name,
            label=name,
            aliases={name, ticker, name.split()[0]},  # "Gold" from "Gold Futures"
            query_hint=f"{name} price trend outlook",
        ))
    # Free-text commodities from the manifest (e.g. "copper", "Gold (GC=F)").
    for raw in manifest.get("commodities", []) or []:
        label = str(raw).strip()
        if label:
            cands.append(Subdomain(
                key=label.lower(), label=label, aliases={label, label.split()[0]},
                query_hint=f"{label} price outlook",
            ))
    return cands


def _distributors_candidates(manifest: dict, masterdata) -> list[Subdomain]:
    cands: list[Subdomain] = []
    for d in masterdata.get_distributors():
        name = d.get("name", "")
        if not name:
            continue
        cands.append(Subdomain(
            key=name,
            label=name,
            aliases={name, d.get("parent", "")},
            query_hint=f"{name} dealer distribution channel",
        ))
    return cands


def _company_candidates(
    entries: list[dict], manifest_key: str, manifest: dict, query_suffix: str
) -> list[Subdomain]:
    """Build company Subdomain candidates from a master-data list plus any
    research-surfaced names in manifest[manifest_key]."""
    seen: set[str] = set()
    cands: list[Subdomain] = []
    for entry in entries:
        name = entry.get("name", "")
        ticker = str(entry.get("ticker", "") or "")
        if name and name not in seen:
            seen.add(name)
            cands.append(Subdomain(
                key=ticker.upper() or name,
                label=name,
                aliases={name, ticker},
                query_hint=f"{name} {query_suffix}",
            ))
    for c in manifest.get(manifest_key, []) or []:
        label = str(c).strip()
        if label and label not in seen:
            seen.add(label)
            cands.append(Subdomain(
                key=label, label=label, aliases={label},
                query_hint=f"{label} {query_suffix}",
            ))
    return cands


def _mining_operators_candidates(manifest: dict, masterdata) -> list[Subdomain]:
    return _company_candidates(
        masterdata.get_operators(), "operators", manifest, "capex equipment spend"
    )


def _construction_companies_candidates(manifest: dict, masterdata) -> list[Subdomain]:
    return _company_candidates(
        masterdata.get_construction(), "construction", manifest, "project pipeline capex"
    )


def _specialized_customers_candidates(manifest: dict, masterdata) -> list[Subdomain]:
    return _company_candidates(
        masterdata.get_others(), "others", manifest, "capital plan equipment"
    )


def _macroeconomics_candidates(manifest: dict, masterdata) -> list[Subdomain]:
    # Countries/regions are surfaced by research (manifest), not master data.
    cands: list[Subdomain] = []
    for region in manifest.get("regions", []) or []:
        label = str(region).strip()
        if label:
            cands.append(Subdomain(
                key=label.lower(), label=label, aliases={label},
                query_hint=f"{label} macroeconomic outlook GDP construction",
            ))
    return cands


def _mining_projects_candidates(manifest: dict, masterdata) -> list[Subdomain]:
    cands: list[Subdomain] = []
    # Mine sites are surfaced by research (manifest), not master data.
    for site in manifest.get("mine_sites", []) or []:
        label = str(site).strip()
        if label:
            cands.append(Subdomain(
                key=label, label=label, aliases={label},
                query_hint=f"{label} mine development",
            ))
    return cands


def _theme_candidates(
    domain: str, merged_chapter: MergedChapter, demand_side: list[str] | None = None
) -> tuple[list[Subdomain], dict]:
    """LLM-derived themes for domains with no fixed master-data entity list.

    Returns ([], {}) when evidence is too thin to bother calling the LLM, and
    ([], usage) when the LLM call fails, returns malformed JSON, or yields
    fewer than 2 themes — the caller's degenerate-entity check (`>= 2`) also
    re-applies this after the alias-presence filter, so this function does not
    need to duplicate that threshold strictly, but short-circuits cheaply here.
    """
    word_count = len(merged_chapter.text.split())
    if word_count < settings.synthesis.theme_extraction_min_evidence_words:
        return [], {}

    evidence_text = _evidence_text(merged_chapter)[
        : settings.synthesis.theme_extraction_max_evidence_chars
    ]
    hint = ""
    if demand_side:
        hint = (
            "\nNote: these third-party demand-side companies were researched for their "
            "commodity demand — if they appear in the evidence above, surface a distinct "
            "third-party theme covering them: " + ", ".join(demand_side) + "\n"
        )
    prompt = THEME_EXTRACTION_TEMPLATE.format(
        domain=domain, evidence_text=evidence_text, hint=hint
    )

    llm = LLMClient()
    resp = llm.complete(
        [{"role": "user", "content": prompt}],
        temperature=settings.synthesis.theme_extraction_temperature,
    )
    usage = llm_usage(resp.usage)

    data = _load_json(resp.content or "")
    themes = data.get("themes") if isinstance(data, dict) else None
    if not isinstance(themes, list):
        return [], usage

    candidates: list[Subdomain] = []
    for t in themes:
        if not isinstance(t, dict):
            continue
        label = str(t.get("label", "")).strip()
        if not label:
            continue
        aliases = {str(a).strip() for a in (t.get("aliases") or []) if str(a).strip()}
        candidates.append(Subdomain(
            key=_slugify(label),
            label=label,
            aliases=aliases | {label},
            query_hint=f"{label} ({domain})",
        ))
    return candidates, usage


def _load_json(raw: str) -> dict | None:
    """Parse a JSON object from an LLM response, tolerating markdown code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("```")
        ).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or label.lower()


_CANDIDATE_BUILDERS = {
    "commodities": _commodities_candidates,
    "competition": _competition_candidates,
    "mining_operators": _mining_operators_candidates,
    "construction_companies": _construction_companies_candidates,
    "specialized_customers": _specialized_customers_candidates,
    "distributors": _distributors_candidates,
    "mining_projects": _mining_projects_candidates,
    "macroeconomics": _macroeconomics_candidates,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evidence_text(merged_chapter: MergedChapter) -> str:
    """Lowercased blob of all evidence in a chapter, for presence checks."""
    parts = [merged_chapter.text]
    for k, v in merged_chapter.figures.items():
        parts.append(f"{k} {v}")
    parts.extend(f"{c.get('title', '')} {c.get('publisher', '')}" for c in merged_chapter.citations)
    try:
        parts.append(json.dumps(merged_chapter.datasets, default=str))
    except Exception:
        pass
    return " ".join(parts).lower()


def _dataset_mentions(dataset: dict, aliases: set[str]) -> bool:
    try:
        blob = json.dumps(dataset, default=str).lower()
    except Exception:
        blob = str(dataset).lower()
    return _mentions(blob, aliases)


def _dataset_identifiers(dataset: dict) -> list[str]:
    """Candidate ticker/symbol/name strings for *dataset*, for master-data lookup.

    series_id is "tool:TICKER[:period]" (see agents/base_domain_agent.py), the
    most reliable identifier; label is "TICKER period" or a bare symbol/name as
    a fallback when series_id is absent.
    """
    idents: list[str] = []
    series_id = str(dataset.get("series_id") or "")
    parts = series_id.split(":")
    if len(parts) >= 2 and parts[1]:
        idents.append(parts[1])
    label = str(dataset.get("label") or "").strip()
    if label:
        idents.append(label.split()[0])
    return idents


def _mentions(text: str, aliases: set[str]) -> bool:
    """True if any alias (>=2 chars) appears in *text* (lowercased) on word boundaries.

    Boundaries are alphanumeric-only (not `\\w`, which includes "_"), so a
    ticker like "CAT" still matches inside a figure key such as "CAT_revenue_bn"
    while a short ticker like "DE" does not false-match inside "expanded".
    Lookarounds (rather than `\\b`) handle aliases containing punctuation such
    as "SAND.ST" or "VOLV-B.ST".
    """
    for alias in aliases:
        a = alias.strip().lower()
        if len(a) < 2:
            continue
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(a)}(?![A-Za-z0-9])", text):
            return True
    return False
