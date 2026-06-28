"""Unit tests for build_collection_plan: the live stem→leaf collection tree with
per-tool-call status that the frontend renders during collection / at Gate 2."""
import json

from memory.sqlite_store import build_collection_plan, _compute_collect_progress

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


def _progress(label="Collecting intelligence from data sources…", event_id=99):
    return {
        "id": event_id,
        "event_type": "progress",
        "stage": "collect",
        "domain": "",
        "label": label,
        "detail": None,
    }


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


def test_trailing_collect_progress_does_not_reset_gate2_statuses():
    events = [
        _event("tool_call", "competition", f"{PLAN_ID}:competition:0"),
        _event("tool_failed_final", "competition", f"{PLAN_ID}:competition:1", error="no data"),
        _event("tool_call", "mining_operators", f"{PLAN_ID}:mining_operators:0"),
        _progress(event_id=500),
    ]

    tree = build_collection_plan(_plan(), events)

    assert tree["total"] == 5
    assert tree["succeeded"] == 2
    assert tree["failed"] == 1
    assert tree["pending"] == 2


def test_collect_progress_total_comes_from_plan_not_domain_filter_events():
    events = [
        {
            "id": 1,
            "event_type": "domain_filter",
            "stage": "collect",
            "domain": "competition",
            "label": "competition: 999/999 calls matched",
            "detail": json.dumps({"matched": 999}),
        },
        _event("tool_call", "competition", f"{PLAN_ID}:competition:0"),
        _event("tool_failed_final", "competition", f"{PLAN_ID}:competition:1", error="no data"),
    ]

    progress = _compute_collect_progress(events, "collect", _plan())

    assert progress["tool_calls_total"] == 5
    assert progress["tool_calls_completed"] == 2
    assert progress["tool_calls_succeeded"] == 1
    assert progress["tool_calls_failed"] == 1


def test_latest_terminal_status_wins_for_same_call_id():
    call_id = f"{PLAN_ID}:competition:0"
    events = [
        _event("tool_call", "competition", call_id),
        _event("tool_failed_final", "competition", call_id, error="no usable data"),
    ]

    tree = build_collection_plan(_plan(), events)
    comp = _by_domain(tree, "competition")
    cat = next(lf for lf in comp["leaves"] if lf["label"] == "Caterpillar Inc.")
    call = next(tc for tc in cat["tool_calls"] if tc["tool"] == "get_company_financials")

    assert call["status"] == "failed"
    assert call["reason"] == "no usable data"


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


def _event_meta(event_type, domain, call_id, data_type=None, count=0, error=""):
    detail = {"call_id": call_id, "tool": "x"}
    if data_type is not None:
        detail["data_type"] = data_type
        detail["count"] = count
    if error:
        detail["error"] = error
    return {"id": 10, "event_type": event_type, "stage": "collect",
            "domain": domain, "label": "", "detail": json.dumps(detail)}


def test_preliminary_plan_renders_leaves_without_tool_calls():
    # A plan emitted mid-planning has leaves but no tool calls yet; the tree should
    # still render the domain→leaf skeleton (all pending) so it builds up in the UI.
    plan = {
        "plan_id": PLAN_ID,
        "leaves": [
            {"key": "CAT", "label": "Caterpillar Inc.", "leaf_type": "company",
             "domain": "competition", "params": {"ticker": "CAT"}},
            {"key": "BHP", "label": "BHP", "leaf_type": "company",
             "domain": "mining_operators", "params": {"ticker": "BHP"}},
        ],
        "planned_tool_calls": [],
    }
    tree = build_collection_plan(plan, [])
    assert tree is not None
    assert tree["total"] == 0 and tree["pending"] == 0
    comp = _by_domain(tree, "competition")
    assert [lf["label"] for lf in comp["leaves"]] == ["Caterpillar Inc."]
    assert comp["leaves"][0]["tool_calls"] == []
    # Empty plan (no leaves and no calls) still renders nothing.
    assert build_collection_plan({"plan_id": "x", "planned_tool_calls": []}, []) is None


def test_succeeded_call_carries_data_type_and_count():
    events = [
        _event_meta("tool_call", "competition", f"{PLAN_ID}:competition:0",
                    data_type="financials", count=42),
    ]
    tree = build_collection_plan(_plan(), events)
    comp = _by_domain(tree, "competition")
    cat = next(lf for lf in comp["leaves"] if lf["label"] == "Caterpillar Inc.")
    call = next(tc for tc in cat["tool_calls"] if tc["tool"] == "get_company_financials")
    assert call["status"] == "succeeded"
    assert call["data_type"] == "financials"
    assert call["count"] == 42
