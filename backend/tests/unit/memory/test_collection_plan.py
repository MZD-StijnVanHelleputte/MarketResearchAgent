"""Unit tests for build_collection_plan: the live stem→leaf collection tree with
per-tool-call status that the frontend renders during collection / at Gate 2."""
import json

from memory.sqlite_store import build_collection_plan

PLAN_ID = "consolidated-run1"


def _plan() -> dict:
    return {
        "plan_id": PLAN_ID,
        "leaves": [
            {"key": "CAT", "label": "Caterpillar Inc.", "leaf_type": "company",
             "domain": "competition", "params": {"ticker": "CAT"}},
            {"key": "BHP", "label": "BHP", "leaf_type": "company",
             "domain": "mining_operators", "params": {"ticker": "BHP"}},
        ],
        "planned_tool_calls": [
            # competition / CAT — 2 calls
            {"tool": "get_company_financials", "domain": "competition", "params": {"ticker": "CAT"}},
            {"tool": "news_search", "domain": "competition", "params": {"query": "Caterpillar Inc. earnings"}},
            # mining_operators / BHP — 2 calls
            {"tool": "get_equity_price", "domain": "mining_operators", "params": {"ticker": "BHP"}},
            {"tool": "get_cash_flow", "domain": "mining_operators", "params": {"ticker": "BHP"}},
            # an unattributable competition call → General bucket
            {"tool": "web_search", "domain": "competition", "params": {"query": "heavy equipment market"}},
        ],
    }


def _event(event_type, domain, call_id, error=""):
    detail = {"call_id": call_id, "tool": "x"}
    if error:
        detail["error"] = error
    return {"id": 10, "event_type": event_type, "stage": "collect",
            "domain": domain, "label": "", "detail": json.dumps(detail)}


def _by_domain(tree, domain):
    return next(d for d in tree["domains"] if d["domain"] == domain)


def test_none_when_no_calls():
    assert build_collection_plan(None, []) is None
    assert build_collection_plan({"plan_id": "x", "planned_tool_calls": []}, []) is None


def test_all_pending_before_any_events():
    tree = build_collection_plan(_plan(), [])
    assert tree["total"] == 5
    assert tree["pending"] == 5 and tree["succeeded"] == 0 and tree["failed"] == 0
    comp = _by_domain(tree, "competition")
    # CAT (financials + news) and a General bucket (web_search).
    labels = {lf["label"] for lf in comp["leaves"]}
    assert "Caterpillar Inc." in labels and "General" in labels
    cat = next(lf for lf in comp["leaves"] if lf["label"] == "Caterpillar Inc.")
    assert cat["total"] == 2 and cat["pending"] == 2


def test_status_maps_via_reconstructed_call_id():
    # competition idx 0 (CAT financials) succeeds; idx 1 (CAT news) fails;
    # mining_operators idx 0 (BHP equity) succeeds.
    events = [
        _event("tool_call", "competition", f"{PLAN_ID}:competition:0"),
        _event("tool_failed_final", "competition", f"{PLAN_ID}:competition:1", error="429 rate limited"),
        _event("tool_call", "mining_operators", f"{PLAN_ID}:mining_operators:0"),
    ]
    tree = build_collection_plan(_plan(), events)
    assert tree["succeeded"] == 2 and tree["failed"] == 1 and tree["pending"] == 2

    comp = _by_domain(tree, "competition")
    assert comp["succeeded"] == 1 and comp["failed"] == 1
    cat = next(lf for lf in comp["leaves"] if lf["label"] == "Caterpillar Inc.")
    assert cat["succeeded"] == 1 and cat["failed"] == 1
    statuses = {tc["tool"]: tc["status"] for tc in cat["tool_calls"]}
    assert statuses["get_company_financials"] == "succeeded"
    assert statuses["news_search"] == "failed"
    failed_call = next(tc for tc in cat["tool_calls"] if tc["tool"] == "news_search")
    assert failed_call["reason"] == "429 rate limited"

    mining = _by_domain(tree, "mining_operators")
    assert mining["succeeded"] == 1 and mining["pending"] == 1


def test_query_call_attributed_to_leaf_by_label():
    tree = build_collection_plan(_plan(), [])
    comp = _by_domain(tree, "competition")
    cat = next(lf for lf in comp["leaves"] if lf["label"] == "Caterpillar Inc.")
    # The "Caterpillar Inc. earnings" news_search query is attributed to the CAT leaf.
    assert any(tc["tool"] == "news_search" for tc in cat["tool_calls"])
    # The generic "heavy equipment market" web_search is not.
    general = next(lf for lf in comp["leaves"] if lf["label"] == "General")
    assert [tc["tool"] for tc in general["tool_calls"]] == ["web_search"]


def test_domains_ordered_by_ownership_priority():
    tree = build_collection_plan(_plan(), [])
    domains = [d["domain"] for d in tree["domains"]]
    # competition (priority 2) precedes mining_operators (priority 3).
    assert domains.index("competition") < domains.index("mining_operators")
    assert _by_domain(tree, "competition")["display"] == "Competition"
