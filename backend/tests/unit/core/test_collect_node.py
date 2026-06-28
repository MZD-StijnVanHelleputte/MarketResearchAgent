"""Unit tests for collect_node (Phase 7: domain sub-agent dispatch).

Phase 7 replaced the flat tool loop with per-domain CrewAI sub-agents.
Tests patch core.graph._run_domain_agent to isolate collect_node logic.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.graph import collect_node, data_review_node, AgentState, _gate2_dataset_view
from core.schemas import ChapterDraft


def _masterdata_mock() -> MagicMock:
    md = MagicMock()
    md.get_competitors.return_value = [{"name": "Caterpillar", "ticker": "CAT"}]
    md.get_commodities.return_value = []
    md.get_distributors.return_value = []
    md.get_operators.return_value = []
    md.get_construction.return_value = []
    md.get_others.return_value = []
    md.resolve_entity.return_value = None
    return md


PLAN = {
    "plan_id": "plan_test",
    "feasibility_score": 0.8,
    "domain_activations": {
        "competition": True,
        "distributors": False,
        "customers": False,
        "mining_projects": False,
        "commodities": True,
        "macro_geopolitics": False,
        "general_search": False,
    },
    "tool_calls": [
        {"tool": "news_search", "domain": "competition", "arguments": {"query": "CAT"}},
        {"tool": "news_search", "domain": "commodities", "arguments": {"query": "copper"}},
    ],
    "rationale": "Test plan",
}


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "run_id": "test-run",
        "session_id": "test-session",
        "user_query": "test query",
        "plans": [PLAN],
        "collection_manifest": {},
        "chapter_sets": {},
        "synthesis_chapters": [],
        "merge_log": [],
        "confidence": 0.0,
        "react_iterations": 0,
        "context_messages": [],
        "warnings": [],
        "stage": "collect",
        "error": None,
    }
    state.update(overrides)
    return state


def _draft(domain: str, plan_id: str = "plan_test") -> ChapterDraft:
    return ChapterDraft(
        domain=domain,
        plan_id=plan_id,
        text=f"Chapter text for {domain}",
        figures={f"{domain}_metric": "100"},
    )


async def _fake_run_domain_agent(plan, domain, run_id, retriever, chunker, collection):
    return plan.get("plan_id", ""), domain, _draft(domain, plan.get("plan_id", ""))


@pytest.mark.asyncio
async def test_collect_node_all_succeed():
    with patch("core.graph._run_domain_agent", side_effect=_fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state())

    assert result["confidence"] == 1.0
    assert result["stage"] == "collect"
    assert "plan_test::competition" in result["chapter_sets"]
    assert "plan_test::commodities" in result["chapter_sets"]


@pytest.mark.asyncio
async def test_collect_node_one_fails():
    call_count = 0

    async def flaky_agent(plan, domain, run_id, retriever, chunker, collection):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("agent crash")
        return plan.get("plan_id", ""), domain, _draft(domain)

    with patch("core.graph._run_domain_agent", side_effect=flaky_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state())

    assert result["confidence"] == 0.5


@pytest.mark.asyncio
async def test_collect_node_no_plan():
    result = await collect_node(_base_state(plans=[]))
    assert result["stage"] == "error"
    assert result["error"] is not None


@pytest.mark.asyncio
async def test_collect_node_all_fail_confidence_zero():
    async def always_fail(plan, domain, *args, **kwargs):
        raise RuntimeError("fail")

    with patch("core.graph._run_domain_agent", side_effect=always_fail), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state())

    assert result["confidence"] == 0.0


@pytest.mark.asyncio
async def test_collect_node_populates_chapter_sets():
    with patch("core.graph._run_domain_agent", side_effect=_fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state())

    key = "plan_test::competition"
    assert key in result["chapter_sets"]
    draft_dict = result["chapter_sets"][key]
    assert draft_dict["domain"] == "competition"
    assert draft_dict["text"]


def test_gate2_dataset_view_trims_rows_and_large_summary():
    summary = "x" * 3000
    dataset = {
        "kind": "summary",
        "summary": summary,
        "rows": [[str(i)] for i in range(10)],
    }

    view = _gate2_dataset_view(dataset)

    assert view is not dataset
    assert view["rows"] == dataset["rows"][:5]
    assert view["rows_truncated"] is True
    assert len(view["summary"]) < len(summary)
    assert "Preview truncated for Gate 2" in view["summary"]
    assert view["summary_truncated"] is True
    assert dataset["summary"] == summary
    assert len(dataset["rows"]) == 10


@pytest.mark.asyncio
async def test_collect_node_no_active_domains():
    """A plan with all domains False should produce an error."""
    empty_plan = dict(PLAN)
    empty_plan["domain_activations"] = {d: False for d in PLAN["domain_activations"]}
    with patch("core.graph._run_domain_agent", side_effect=_fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state(plans=[empty_plan]))

    assert result["stage"] == "error"


@pytest.mark.asyncio
async def test_collect_node_reuses_prior_successful_draft():
    """On a backtrack, only domains without a good prior draft are re-run."""
    ran: list[str] = []

    async def tracking_agent(plan, domain, run_id, retriever, chunker, collection):
        ran.append(domain)
        return plan.get("plan_id", ""), domain, _draft(domain, plan.get("plan_id", ""))

    # competition already has a good draft from a previous pass; commodities does not.
    prior = {
        "plan_test::competition": _draft("competition").model_dump(),
    }
    state = _base_state(chapter_sets=prior)
    with patch("core.graph._run_domain_agent", side_effect=tracking_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(state)

    assert ran == ["commodities"]  # competition reused, not re-run
    assert result["confidence"] == 1.0
    assert "plan_test::competition" in result["chapter_sets"]
    assert "plan_test::commodities" in result["chapter_sets"]


@pytest.mark.asyncio
async def test_collect_node_consolidated_plan_shape():
    """A ConsolidatedPlan dict (domains_active: list) must activate domains."""
    consolidated_plan = {
        "plan_id": "consolidated_test",
        "domains_active": ["competition", "commodities"],
        "planned_tool_calls": [
            {"tool": "news_search", "domain": "competition", "params": {"query": "CAT"}},
            {"tool": "news_search", "domain": "commodities", "params": {"query": "copper"}},
        ],
        "rationale": "Test consolidated plan",
    }
    with patch("core.graph._run_domain_agent", side_effect=_fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(_base_state(plans=[consolidated_plan]))

    assert result["stage"] != "error"
    assert result["confidence"] == 1.0
    assert "consolidated_test::competition" in result["chapter_sets"]
    assert "consolidated_test::commodities" in result["chapter_sets"]


@pytest.mark.asyncio
async def test_data_review_node_groups_by_entity_and_surfaces_failures():
    """Gate 2 payload nests datasets under the entity they describe and carries
    structured failed tools so the UI can show 'tried but failed' in red."""
    draft = ChapterDraft(
        domain="competition", plan_id="plan_test", text="CAT update",
        datasets=[{
            "tool": "get_company_financials", "title": "CAT FY — 4 row(s)",
            "kind": "table", "data_type": "financials", "label": "CAT FY", "count": 4,
            "series_id": "get_company_financials:CAT:annual",
        }],
        failed_tools=[{
            "tool": "search_sec_filings", "tool_display": "SEC EDGAR",
            "reason": "HTTP 429 rate limited",
        }],
    )
    state = _base_state(
        chapter_sets={"plan_test::competition": draft.model_dump()},
        plans=[{"plan_id": "plan_test", "entity_manifest": {"tickers": ["CAT"]}}],
    )

    captured: dict = {}

    def fake_interrupt(payload):
        captured["payload"] = payload
        return "approve"

    store = MagicMock()
    store.get_preference = AsyncMock(return_value=None)
    with patch("core.graph.interrupt", side_effect=fake_interrupt), \
         patch("core.graph.MasterDataService", return_value=_masterdata_mock()), \
         patch("core.graph.SqliteStore", return_value=store):
        result = await data_review_node(state)

    assert result["stage"] == "synthesize"
    payload = captured["payload"]
    assert payload["gate"] == 2
    comp = next(d for d in payload["domains"] if d["domain"] == "competition")
    labels = {e["label"] for e in comp["entities"]}
    assert "Caterpillar" in labels
    assert comp["failed_tools"][0]["tool_display"] == "SEC EDGAR"


def test_draft_source_entries_types_datasets_and_failures():
    """Live Sources entries are typed (data_type + label + count); failures are
    flagged in red via failed=True with a reason."""
    from core.graph import _draft_source_entries

    draft = ChapterDraft(
        domain="commodities", plan_id="p", text="x",
        datasets=[{"tool": "get_mining_metals_prices", "title": "COPPER — 250 point(s)",
                   "data_type": "numeric_series", "label": "COPPER", "count": 250}],
        failed_tools=[{"tool": "get_macro_indicator", "tool_display": "Macro (FRED)",
                       "reason": "HTTP 429"}],
    ).model_dump()

    entries = _draft_source_entries("commodities", draft)
    good = next(e for e in entries if not e["failed"])
    bad = next(e for e in entries if e["failed"])

    assert good["data_type"] == "numeric_series"
    assert good["label"] == "COPPER"
    assert good["count"] == 250
    assert bad["data_type"] == "failed"
    assert bad["reason"] == "HTTP 429"
