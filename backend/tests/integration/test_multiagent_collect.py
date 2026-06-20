"""Integration tests for Phase 7 multi-agent collect + merge pipeline.

All domain agent tools are mocked; no real API calls are made.
Tests validate:
  1. All (plan × domain) tasks are scheduled in collect_node.
  2. Merged chapters have no contradicting figures for the same metric.
  3. Diversity recovery appends a supplementary section when overlap > threshold.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.graph import AgentState, collect_node, synthesize_node
from core.schemas import ChapterDraft, MergedChapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_plan(plan_id: str, domains: list[str], feasibility: float = 0.8) -> dict:
    return {
        "plan_id": plan_id,
        "feasibility_score": feasibility,
        "quality_score": 0.8,
        "combined_score": feasibility * 0.6 + 0.8 * 0.4,
        "is_survivor": True,
        "domain_activations": {d: (d in domains) for d in [
            "competition", "distributors", "customers",
            "mining_projects", "commodities", "macro_geopolitics", "general_search"
        ]},
        "tool_calls": [
            {"tool": "news_search", "domain": d, "arguments": {"query": "test"}}
            for d in domains
        ],
        "rationale": "test plan",
    }


def _base_state(plans: list[dict]) -> AgentState:
    return {
        "run_id": "run_test",
        "session_id": "sess_test",
        "user_query": "What is Caterpillar's capex cycle?",
        "plans": plans,
        "collection_manifest": {},
        "chapter_sets": {},
        "synthesis_chapters": [],
        "merge_log": [],
        "confidence": 0.0,
        "react_iterations": 0,
        "context_messages": [],
        "stage": "collect",
        "warnings": [],
        "error": None,
    }


def _draft(domain: str, plan_id: str, text: str = "chapter text",
           figures: dict | None = None) -> ChapterDraft:
    return ChapterDraft(
        domain=domain,
        plan_id=plan_id,
        text=text,
        figures=figures or {},
    )


# ---------------------------------------------------------------------------
# Test 1 — all (plan × domain) tasks are scheduled
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_tasks_scheduled():
    """collect_node must schedule one task per (plan, active_domain) pair."""
    active_domains = ["competition", "commodities"]
    plans = [_make_plan(f"plan_{i}", active_domains) for i in range(3)]
    state = _base_state(plans)

    call_log: list[tuple[str, str]] = []

    async def fake_run_domain_agent(plan, domain, run_id, retriever, chunker, collection):
        call_log.append((plan["plan_id"], domain))
        draft = _draft(domain, plan["plan_id"])
        return plan["plan_id"], domain, draft

    with patch("core.graph._run_domain_agent", side_effect=fake_run_domain_agent), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(state)

    expected = len(plans) * len(active_domains)
    assert len(call_log) == expected, (
        f"Expected {expected} tasks, got {len(call_log)}: {call_log}"
    )
    # Verify correct (plan_id, domain) pairs were called
    for plan in plans:
        for domain in active_domains:
            assert (plan["plan_id"], domain) in call_log


@pytest.mark.asyncio
async def test_collect_node_builds_chapter_sets():
    """collect_node must populate chapter_sets keyed by 'plan_id::domain'."""
    plans = [_make_plan("plan_A", ["competition"])]
    state = _base_state(plans)

    async def fake_run(plan, domain, run_id, retriever, chunker, collection):
        return plan["plan_id"], domain, _draft(domain, plan["plan_id"])

    with patch("core.graph._run_domain_agent", side_effect=fake_run), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(state)

    assert "plan_A::competition" in result["chapter_sets"]
    assert result["confidence"] == 1.0


@pytest.mark.asyncio
async def test_collect_node_handles_agent_exceptions():
    """collect_node must gracefully handle exceptions from individual domain agents."""
    plans = [_make_plan("plan_A", ["competition", "commodities"])]
    state = _base_state(plans)
    call_count = 0

    async def fake_run(plan, domain, run_id, retriever, chunker, collection):
        nonlocal call_count
        call_count += 1
        if domain == "competition":
            raise RuntimeError("agent crash")
        return plan["plan_id"], domain, _draft(domain, plan["plan_id"])

    with patch("core.graph._run_domain_agent", side_effect=fake_run), \
         patch("core.graph.Retriever"), \
         patch("core.graph.Chunker"):
        result = await collect_node(state)

    assert call_count == 2  # both tasks were attempted
    assert "plan_A::commodities" in result["chapter_sets"]
    assert result["confidence"] == 0.5  # 1/2 succeeded


# ---------------------------------------------------------------------------
# Test 2 — merged chapters have no contradicting figures for same metric
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_merged_chapters_no_contradicting_figures():
    """After merge, same metric must not appear with conflicting values in output."""
    plans = [
        _make_plan("plan_A", ["competition"], feasibility=0.9),
        _make_plan("plan_B", ["competition"], feasibility=0.7),
    ]
    state = _base_state(plans)
    # Both plans report same metric with same value → no contradiction
    state["chapter_sets"] = {
        "plan_A::competition": _draft("competition", "plan_A", figures={"CAT_rev": "$64B"}).model_dump(),
        "plan_B::competition": _draft("competition", "plan_B", figures={"CAT_rev": "$64B"}).model_dump(),
    }

    mock_synth = MagicMock()
    mock_synth.run.return_value = {"domain": "competition", "text": "polished text"}
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ([], [])

    with patch("core.graph.SynthesisAgent", return_value=mock_synth), \
         patch("core.graph.Retriever", return_value=mock_retriever):
        result = await synthesize_node(state)

    # merge_log should be empty (no contradictions)
    assert result["merge_log"] == []
    assert len(result["synthesis_chapters"]) == 1
    assert result["synthesis_chapters"][0]["domain"] == "competition"


@pytest.mark.asyncio
async def test_merged_chapters_logs_contradictions():
    """Contradicting figures across plans must be logged in merge_log."""
    plans = [
        _make_plan("plan_A", ["competition"], feasibility=0.9),
        _make_plan("plan_B", ["competition"], feasibility=0.7),
    ]
    state = _base_state(plans)
    state["chapter_sets"] = {
        "plan_A::competition": _draft("competition", "plan_A", figures={"CAT_rev": "$64B"}).model_dump(),
        "plan_B::competition": _draft("competition", "plan_B", figures={"CAT_rev": "$50B"}).model_dump(),
    }

    mock_synth = MagicMock()
    mock_synth.run.return_value = {"domain": "competition", "text": "polished"}
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ([], [])

    with patch("core.graph.SynthesisAgent", return_value=mock_synth), \
         patch("core.graph.Retriever", return_value=mock_retriever):
        result = await synthesize_node(state)

    assert len(result["merge_log"]) >= 1
    assert "CAT_rev" in result["merge_log"][0]


# ---------------------------------------------------------------------------
# Test 3 — diversity recovery appends supplementary section when overlap > threshold
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recovery_appends_supplementary_when_high_overlap():
    """When Jaccard overlap exceeds threshold, a supplementary section is appended."""
    from config import settings

    plans = [_make_plan("plan_A", ["competition"], feasibility=0.9)]
    state = _base_state(plans)
    # Chapter text is identical → Jaccard overlap will be 1.0
    state["chapter_sets"] = {
        "plan_A::competition": _draft(
            "competition", "plan_A",
            text="copper mining equipment komatsu caterpillar revenue capex"
        ).model_dump(),
    }

    # Pruned plan on the state bus (structurally distinct: activates commodities, not competition)
    pruned_plan = _make_plan("plan_pruned", ["commodities"])
    pruned_plan["is_survivor"] = False

    recovery_draft = _draft("commodities", "plan_pruned", text="Recovery commodities text")

    # Mock agent class that returns the recovery draft
    mock_agent_instance = MagicMock()
    mock_agent_instance.run = AsyncMock(return_value=recovery_draft)
    mock_agent_cls = MagicMock(return_value=mock_agent_instance)

    mock_domain_agents = {"commodities": mock_agent_cls}

    mock_synth = MagicMock()
    mock_synth.run.side_effect = lambda domain, mc, *a, **kw: {"domain": domain, "text": f"synth {domain}"}
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = ([], [])

    with patch("core.graph.SynthesisAgent", return_value=mock_synth), \
         patch("core.graph.Retriever", return_value=mock_retriever), \
         patch("core.graph.get_all_plans_direct", return_value=[plans[0], pruned_plan]), \
         patch("core.graph.chapter_set_overlap", return_value=settings.tot.diversity_overlap_threshold + 0.05), \
         patch("agents.DOMAIN_AGENTS", mock_domain_agents):
        result = await synthesize_node(state)

    domains_in_output = [ch["domain"] for ch in result["synthesis_chapters"]]
    assert "supplementary" in domains_in_output, (
        f"Expected supplementary section but got domains: {domains_in_output}"
    )
    # A warning should have been recorded
    assert any("recovery" in w.lower() or "overlap" in w.lower() for w in result["warnings"])
