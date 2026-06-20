"""Chapter merger: combines parallel ChapterDrafts from multiple survivor plans.

For each domain, three survivor plans each produce a ChapterDraft. This module:
  1. Groups drafts by domain.
  2. Picks the authoritative draft (highest-feasibility plan).
  3. Detects figure contradictions across lower-feasibility plans and logs them.
  4. Merges citations (deduplicated).
  5. Exposes chapter_set_overlap() for the diversity recovery check.
"""
import logging
import re
from collections import defaultdict

from core.schemas import ChapterDraft, MergedChapter

logger = logging.getLogger(__name__)


def merge_chapter_sets(
    chapter_sets: dict[str, dict],
    plans: list[dict],
) -> tuple[list[MergedChapter], list[str]]:
    """Merge parallel chapter drafts into one MergedChapter per domain.

    Args:
        chapter_sets: {"plan_id::domain": ChapterDraft.model_dump()}
        plans: survivor plan dicts sorted by combined_score descending.

    Returns:
        (merged_chapters, merge_log)
    """
    # Build feasibility lookup {plan_id: feasibility_score}
    feasibility: dict[str, float] = {
        p.get("plan_id", ""): float(p.get("feasibility_score", 0.0))
        for p in plans
    }

    # Group drafts by domain: {domain: [(feasibility, plan_id, ChapterDraft)]}
    by_domain: dict[str, list[tuple[float, str, ChapterDraft]]] = defaultdict(list)
    for key, draft_dict in chapter_sets.items():
        if "::" not in key:
            continue
        plan_id, domain = key.split("::", 1)
        draft = ChapterDraft.model_validate(draft_dict)
        score = feasibility.get(plan_id, 0.0)
        by_domain[domain].append((score, plan_id, draft))

    merge_log: list[str] = []
    merged_chapters: list[MergedChapter] = []

    for domain, entries in by_domain.items():
        # Sort descending by feasibility; authoritative = highest-feasibility plan
        entries.sort(key=lambda e: e[0], reverse=True)
        auth_score, auth_plan_id, auth_draft = entries[0]

        # Start with authoritative figures
        figures: dict[str, str] = dict(auth_draft.figures)
        citations: list[str] = list(auth_draft.citations)
        contradiction_flags: list[str] = list(auth_draft.contradiction_flags)
        source_plan_ids: list[str] = [auth_plan_id]
        # Union datasets across all survivor drafts for this domain (deduped by
        # series_id where available, since titles embed a row count that varies
        # by plan/time window — falls back to tool+title for list/summary kinds)
        # so tables collected by any plan reach the report, not just whatever the
        # highest-feasibility plan happened to call.
        seen_datasets: set = set()
        datasets: list[dict] = []
        for _, _, entry_draft in entries:
            for ds in entry_draft.datasets:
                key = ds.get("series_id") or (ds.get("tool", ""), ds.get("title", ""))
                if key not in seen_datasets:
                    seen_datasets.add(key)
                    datasets.append(ds)

        # Check lower-priority drafts for contradictions and additional citations
        for alt_score, alt_plan_id, alt_draft in entries[1:]:
            source_plan_ids.append(alt_plan_id)
            for metric, alt_value in alt_draft.figures.items():
                if metric in figures:
                    auth_value = figures[metric]
                    if _figures_contradict(auth_value, alt_value):
                        note = (
                            f"[{domain}] '{metric}': using '{auth_value}' "
                            f"from {auth_plan_id} (feasibility={auth_score:.2f}); "
                            f"discarding '{alt_value}' from {alt_plan_id} "
                            f"(feasibility={alt_score:.2f})"
                        )
                        merge_log.append(note)
                        contradiction_flags.append(note)
                        logger.debug("merger: %s", note)
                else:
                    # Metric only in lower-priority plan — include it
                    figures[metric] = alt_value

            # Merge citations (deduplicated)
            for cit in alt_draft.citations:
                if cit not in citations:
                    citations.append(cit)

            # Merge contradiction_flags from sub-agents
            for flag in alt_draft.contradiction_flags:
                if flag not in contradiction_flags:
                    contradiction_flags.append(flag)

        merged_chapters.append(MergedChapter(
            domain=domain,
            text=auth_draft.text,
            figures=figures,
            citations=citations,
            contradiction_flags=contradiction_flags,
            source_plan_ids=source_plan_ids,
            datasets=datasets,
        ))

    if merge_log:
        logger.info("merger: %d contradiction(s) resolved across %d domains",
                    len(merge_log), len(merged_chapters))

    return merged_chapters, merge_log


def chapter_set_overlap(chapters: list[MergedChapter]) -> float:
    """Compute average pairwise Jaccard overlap across all merged chapters.

    Uses the union of (figures keys) + (significant tokens from text) as the
    entity set for each chapter. Returns 0.0 if fewer than 2 chapters.
    """
    if len(chapters) < 2:
        return 0.0

    token_sets = [_entity_tokens(ch) for ch in chapters]
    pairs = [
        (i, j)
        for i in range(len(token_sets))
        for j in range(i + 1, len(token_sets))
    ]
    scores: list[float] = []
    for i, j in pairs:
        intersection = len(token_sets[i] & token_sets[j])
        union = len(token_sets[i] | token_sets[j])
        scores.append(intersection / union if union > 0 else 0.0)

    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _figures_contradict(val_a: str, val_b: str) -> bool:
    """Return True if two figure value strings represent meaningfully different numbers.

    Uses a 5% relative tolerance; returns True on non-numeric mismatch.
    """
    a = _extract_number(val_a)
    b = _extract_number(val_b)
    if a is None or b is None:
        return val_a.strip().lower() != val_b.strip().lower()
    denom = max(abs(a), abs(b))
    if denom == 0:
        return False
    return abs(a - b) / denom > 0.05


def _extract_number(s: str) -> float | None:
    """Extract the first numeric value from a string (strips commas, currency symbols)."""
    cleaned = s.replace(",", "").replace("$", "").replace("€", "").replace("£", "")
    m = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    return float(m.group()) if m else None


def _entity_tokens(chapter: MergedChapter) -> set[str]:
    """Build a token set from figure keys + significant words from text."""
    tokens: set[str] = set(chapter.figures.keys())
    # Add words > 5 chars from text as a simple named-entity proxy
    for word in chapter.text.split():
        clean = re.sub(r"[^a-zA-Z0-9]", "", word).lower()
        if len(clean) > 5:
            tokens.add(clean)
    return tokens
