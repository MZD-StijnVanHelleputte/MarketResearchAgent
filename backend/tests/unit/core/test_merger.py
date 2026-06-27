"""Unit tests for core/merger.py."""
import pytest

from core.merger import (
    _figures_contradict,
    assign_global_citation_ids,
    chapter_set_overlap,
    merge_chapter_sets,
)
from core.schemas import ChapterDraft, MergedChapter


def _citation(url: str) -> dict:
    return {"id": None, "title": url, "url": url, "publisher": None}


def _make_draft(domain: str, plan_id: str, figures: dict | None = None,
                citations: list[dict] | None = None, text: str = "some text",
                datasets: list[dict] | None = None) -> dict:
    return ChapterDraft(
        domain=domain,
        plan_id=plan_id,
        text=text,
        figures=figures or {},
        citations=citations or [],
        datasets=datasets or [],
    ).model_dump()


def _make_plan(plan_id: str, feasibility: float) -> dict:
    return {"plan_id": plan_id, "feasibility_score": feasibility,
            "domain_activations": {"competition": True}}


# ---------------------------------------------------------------------------
# merge_chapter_sets
# ---------------------------------------------------------------------------

def test_merge_uses_highest_feasibility_figures():
    chapter_sets = {
        "plan_A::competition": _make_draft("competition", "plan_A",
                                           figures={"CAT_revenue": "$64B"}),
        "plan_B::competition": _make_draft("competition", "plan_B",
                                           figures={"CAT_revenue": "$63B"}),
    }
    plans = [
        _make_plan("plan_A", feasibility=0.9),
        _make_plan("plan_B", feasibility=0.7),
    ]
    merged, log = merge_chapter_sets(chapter_sets, plans)
    assert len(merged) == 1
    mc = merged[0]
    assert mc.domain == "competition"
    # plan_A has higher feasibility → its figure wins
    assert mc.figures["CAT_revenue"] == "$64B"


def test_merge_logs_contradictions():
    chapter_sets = {
        "plan_A::competition": _make_draft("competition", "plan_A",
                                           figures={"revenue": "$64B"}),
        "plan_B::competition": _make_draft("competition", "plan_B",
                                           figures={"revenue": "$50B"}),  # >5% diff
    }
    plans = [_make_plan("plan_A", 0.9), _make_plan("plan_B", 0.7)]
    merged, log = merge_chapter_sets(chapter_sets, plans)
    assert len(log) >= 1
    assert "revenue" in log[0]
    assert merged[0].contradiction_flags  # flag propagated to merged chapter


def test_merge_no_contradiction_within_tolerance():
    chapter_sets = {
        "plan_A::competition": _make_draft("competition", "plan_A",
                                           figures={"price": "$10.00"}),
        "plan_B::competition": _make_draft("competition", "plan_B",
                                           figures={"price": "$10.02"}),  # < 5% diff
    }
    plans = [_make_plan("plan_A", 0.9), _make_plan("plan_B", 0.7)]
    _, log = merge_chapter_sets(chapter_sets, plans)
    # 0.02 / 10.00 = 0.2% — within 5% tolerance, no contradiction
    assert log == []


def test_merge_combines_citations_deduplicated():
    chapter_sets = {
        "plan_A::competition": _make_draft("competition", "plan_A",
                                           citations=[_citation("https://a.com"), _citation("https://b.com")]),
        "plan_B::competition": _make_draft("competition", "plan_B",
                                           citations=[_citation("https://b.com"), _citation("https://c.com")]),
    }
    plans = [_make_plan("plan_A", 0.9), _make_plan("plan_B", 0.7)]
    merged, _ = merge_chapter_sets(chapter_sets, plans)
    urls = [c["url"] for c in merged[0].citations]
    assert "https://a.com" in urls
    assert "https://b.com" in urls
    assert "https://c.com" in urls
    # no duplicates — the same url cited by both plans collapses to one entry
    assert len(urls) == len(set(urls))


def test_assign_global_citation_ids_shares_id_across_chapters():
    chapters = [
        MergedChapter(domain="competition", text="t", citations=[_citation("https://a.com")]),
        MergedChapter(domain="commodities", text="t", citations=[_citation("https://a.com"), _citation("https://b.com")]),
    ]
    registry = assign_global_citation_ids(chapters)
    cid_a = chapters[0].citations[0]["id"]
    cid_a_again = chapters[1].citations[0]["id"]
    cid_b = chapters[1].citations[1]["id"]
    assert cid_a == cid_a_again
    assert cid_a != cid_b
    assert registry[cid_a]["url"] == "https://a.com"
    assert registry[cid_b]["url"] == "https://b.com"


def test_assign_global_citation_ids_extends_existing_registry():
    first = [MergedChapter(domain="competition", text="t", citations=[_citation("https://a.com")])]
    registry = assign_global_citation_ids(first)
    cid_a = first[0].citations[0]["id"]

    second = [MergedChapter(domain="supplementary", text="t", citations=[_citation("https://a.com"), _citation("https://z.com")])]
    registry = assign_global_citation_ids(second, existing=registry)
    assert second[0].citations[0]["id"] == cid_a  # same source keeps its id
    assert second[0].citations[1]["id"] != cid_a
    assert len(registry) == 2


def test_merge_uses_auth_plan_text():
    chapter_sets = {
        "plan_A::competition": _make_draft("competition", "plan_A", text="AUTH TEXT"),
        "plan_B::competition": _make_draft("competition", "plan_B", text="OTHER TEXT"),
    }
    plans = [_make_plan("plan_A", 0.9), _make_plan("plan_B", 0.7)]
    merged, _ = merge_chapter_sets(chapter_sets, plans)
    assert merged[0].text == "AUTH TEXT"


def test_merge_includes_unique_figures_from_lower_plans():
    chapter_sets = {
        "plan_A::competition": _make_draft("competition", "plan_A",
                                           figures={"CAT_revenue": "$64B"}),
        "plan_B::competition": _make_draft("competition", "plan_B",
                                           figures={"VOL_revenue": "$12B"}),  # unique metric
    }
    plans = [_make_plan("plan_A", 0.9), _make_plan("plan_B", 0.7)]
    merged, _ = merge_chapter_sets(chapter_sets, plans)
    # Both metrics should appear (no conflict)
    assert merged[0].figures.get("CAT_revenue") == "$64B"
    assert merged[0].figures.get("VOL_revenue") == "$12B"


def test_merge_unions_datasets_across_survivors_deduped():
    chapter_sets = {
        "plan_A::competition": _make_draft(
            "competition", "plan_A",
            datasets=[{"tool": "get_equity_history", "title": "CAT — 5y", "kind": "table",
                       "columns": ["date", "close"], "rows": [["2026-01-01", "350"]]}],
        ),
        "plan_B::competition": _make_draft(
            "competition", "plan_B",
            datasets=[
                {"tool": "get_equity_history", "title": "CAT — 5y", "kind": "table",
                 "columns": ["date", "close"], "rows": [["2026-01-01", "350"]]},
                {"tool": "get_equity_history", "title": "DE — 5y", "kind": "table",
                 "columns": ["date", "close"], "rows": [["2026-01-01", "120"]]},
            ],
        ),
    }
    plans = [_make_plan("plan_A", 0.9), _make_plan("plan_B", 0.7)]
    merged, _ = merge_chapter_sets(chapter_sets, plans)
    titles = [ds["title"] for ds in merged[0].datasets]
    assert titles == ["CAT — 5y", "DE — 5y"]


def test_merge_dedupes_datasets_by_series_id_despite_title_drift():
    """Two plans pulling the same series with different row counts have different
    titles but the same series_id — they must still collapse to one dataset."""
    chapter_sets = {
        "plan_A::commodities": _make_draft(
            "commodities", "plan_A",
            datasets=[{"tool": "get_commodity_price", "title": "copper — 120 point(s)",
                       "kind": "table", "columns": ["date", "value (usd)"],
                       "rows": [["2026-01-01", "4.10"]], "series_id": "get_commodity_price:copper"}],
        ),
        "plan_B::commodities": _make_draft(
            "commodities", "plan_B",
            datasets=[{"tool": "get_commodity_price", "title": "copper — 118 point(s)",
                       "kind": "table", "columns": ["date", "value (usd)"],
                       "rows": [["2026-01-03", "4.12"]], "series_id": "get_commodity_price:copper"}],
        ),
    }
    plans = [_make_plan("plan_A", 0.9), _make_plan("plan_B", 0.7)]
    merged, _ = merge_chapter_sets(chapter_sets, plans)
    assert len(merged[0].datasets) == 1
    assert merged[0].datasets[0]["title"] == "copper — 120 point(s)"  # authoritative plan wins


def test_merge_empty_chapter_sets():
    merged, log = merge_chapter_sets({}, [])
    assert merged == []
    assert log == []


# ---------------------------------------------------------------------------
# chapter_set_overlap
# ---------------------------------------------------------------------------

def test_jaccard_identical_chapters():
    ch = MergedChapter(domain="a", text="copper gold iron mining equipment komatsu",
                       figures={"copper_price": "4.12"})
    overlap = chapter_set_overlap([ch, ch])
    assert overlap == 1.0


def test_jaccard_disjoint_chapters():
    ch_a = MergedChapter(domain="a", text="alpha beta gamma delta epsilon",
                         figures={"metric_a": "1"})
    ch_b = MergedChapter(domain="b", text="omega theta lambda sigma upsilon",
                         figures={"metric_b": "2"})
    overlap = chapter_set_overlap([ch_a, ch_b])
    assert overlap == 0.0


def test_jaccard_single_chapter_returns_zero():
    ch = MergedChapter(domain="a", text="some text")
    assert chapter_set_overlap([ch]) == 0.0


def test_jaccard_partial_overlap():
    ch_a = MergedChapter(domain="a", text="copper mining equipment komatsu", figures={})
    ch_b = MergedChapter(domain="b", text="copper mining caterpillar volvo", figures={})
    overlap = chapter_set_overlap([ch_a, ch_b])
    assert 0.0 < overlap < 1.0


# ---------------------------------------------------------------------------
# _figures_contradict
# ---------------------------------------------------------------------------

def test_figures_contradict_large_diff():
    assert _figures_contradict("$64B", "$50B") is True


def test_figures_contradict_small_diff():
    assert _figures_contradict("$10.00", "$10.02") is False


def test_figures_contradict_non_numeric_same():
    assert _figures_contradict("positive", "positive") is False


def test_figures_contradict_non_numeric_different():
    assert _figures_contradict("positive", "negative") is True
