"""Unit tests for synthesize_node (Phase 7: merger + SynthesisAgent).

Phase 7 replaced direct LLMClient calls with:
  1. merge_chapter_sets() from core/merger.py
  2. SynthesisAgent().run() per domain
Tests patch core.graph.SynthesisAgent and core.graph.Retriever.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.graph import synthesize_node, partial_brief_node, AgentState
from core.schemas import ChapterDraft


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
    "tool_calls": [],
    "rationale": "Test",
}


def _chapter_sets(domains: list[str], text: str = "chapter text") -> dict:
    return {
        f"plan_test::{d}": ChapterDraft(
            domain=d, plan_id="plan_test", text=text, figures={}, citations=[]
        ).model_dump()
        for d in domains
    }


def _base_state(**overrides) -> AgentState:
    state: AgentState = {
        "run_id": "test-run",
        "session_id": "test-session",
        "user_query": "test query",
        "plans": [PLAN],
        "collection_manifest": {},
        "chapter_sets": _chapter_sets(["competition", "commodities"]),
        "synthesis_chapters": [],
        "merge_log": [],
        "confidence": 1.0,
        "react_iterations": 0,
        "context_messages": [],
        "warnings": [],
        "stage": "collect",
        "error": None,
    }
    state.update(overrides)
    return state


@pytest.fixture(autouse=True)
def _mock_exec_summary_llm():
    """Keep exec-summary generation hermetic and fast.

    synthesize_node builds a real LLMClient for the executive summary, which would
    otherwise hit the network (and now retry with backoff on failure). Patch it to
    return a canned summary so unit tests stay deterministic and quick. Individual
    tests can override this patch to exercise failure handling.
    """
    resp = MagicMock()
    resp.content = "Canned executive summary sentence. " * 80  # well-formed, non-empty
    resp.usage = {}
    client = MagicMock()
    client.complete.return_value = resp
    with patch("core.graph.LLMClient", return_value=client):
        yield


def _mock_retriever():
    r = MagicMock()
    r.retrieve.return_value = ([], [])
    return r


def _mock_synth(text: str = "chapter text"):
    m = MagicMock()
    m.run.side_effect = lambda domain, mc, *a, **kw: {"domain": domain, "text": text}
    return m


@pytest.mark.asyncio
async def test_synthesize_node_writes_chapters_for_active_domains():
    with patch("core.graph.SynthesisAgent", return_value=_mock_synth("This is the chapter text.")), \
         patch("core.graph.Retriever", return_value=_mock_retriever()):
        result = await synthesize_node(_base_state())

    assert result["stage"] == "done"
    chapters = result["synthesis_chapters"]
    # competition + commodities active
    assert len(chapters) == 2
    domains = {ch["domain"] for ch in chapters}
    assert domains == {"competition", "commodities"}
    for ch in chapters:
        assert ch["text"] == "This is the chapter text."


@pytest.mark.asyncio
async def test_synthesize_node_empty_chapter_sets_uses_fallback():
    """With no chapter_sets, synthesize_node falls back to 'No data collected' MergedChapters."""
    with patch("core.graph.SynthesisAgent", return_value=_mock_synth()), \
         patch("core.graph.Retriever", return_value=_mock_retriever()):
        result = await synthesize_node(_base_state(chapter_sets={}))

    assert result["stage"] == "done"
    # Fallback still produces chapters for active domains
    domains = {ch["domain"] for ch in result["synthesis_chapters"]}
    assert "competition" in domains
    assert "commodities" in domains


@pytest.mark.asyncio
async def test_synthesize_node_no_plan_returns_error():
    result = await synthesize_node(_base_state(plans=[]))
    assert result["stage"] == "error"


@pytest.mark.asyncio
async def test_synthesize_node_includes_exec_summary():
    """The canned exec-summary LLM response flows through to the result."""
    with patch("core.graph.SynthesisAgent", return_value=_mock_synth()), \
         patch("core.graph.Retriever", return_value=_mock_retriever()):
        result = await synthesize_node(_base_state())

    assert result["exec_summary"].startswith("Canned executive summary")


@pytest.mark.asyncio
async def test_synthesize_node_exec_summary_failure_warns_and_falls_back():
    """When exec-summary generation fails after retries, the run still completes,
    a warning is surfaced (not just logged), and an honestly-labelled fallback is used."""
    failing = MagicMock()
    failing.complete.side_effect = RuntimeError("429 rate limit")
    with patch("core.graph.SynthesisAgent", return_value=_mock_synth("Chapter body text.")), \
         patch("core.graph.Retriever", return_value=_mock_retriever()), \
         patch("core.graph.LLMClient", return_value=failing), \
         patch("core.graph.call_with_backoff", side_effect=lambda fn, *a, **k: fn(*a, **k)):
        result = await synthesize_node(_base_state())

    assert result["stage"] == "done"
    assert any("executive summary" in w.lower() for w in result["warnings"])
    assert result["exec_summary"].startswith("_The executive summary could not be generated")


@pytest.mark.asyncio
async def test_synthesize_node_merge_log_in_result():
    """merge_log must appear in the return dict even if empty."""
    with patch("core.graph.SynthesisAgent", return_value=_mock_synth()), \
         patch("core.graph.Retriever", return_value=_mock_retriever()):
        result = await synthesize_node(_base_state())

    assert "merge_log" in result
    assert isinstance(result["merge_log"], list)


def _entity_rich_state() -> AgentState:
    """A competition chapter whose text/figures mention >=2 master-data competitors."""
    draft = ChapterDraft(
        domain="competition",
        plan_id="plan_test",
        text="Caterpillar posted strong Q3 results. Sandvik expanded its mining tools range.",
        figures={"CAT_revenue_bn": "67.1", "SAND.ST_margin": "19%"},
        citations=[],
    ).model_dump()
    plan = dict(PLAN, entity_manifest={"tickers": ["CAT", "SAND.ST"]})
    return _base_state(
        plans=[plan],
        chapter_sets={"plan_test::competition": draft},
    )


def _mock_masterdata():
    md = MagicMock()
    md.get_competitors.return_value = [
        {"name": "Caterpillar", "ticker": "CAT"},
        {"name": "Sandvik", "ticker": "SAND.ST"},
    ]
    return md


def _mock_synth_hierarchical():
    m = MagicMock()
    m.run_subchapter.side_effect = lambda domain, key, label, *a, **kw: {
        "domain": domain,
        "subdomain_key": key,
        "subdomain_label": label,
        "text": f"Leaf analysis for {label}.",
        "usage": {},
    }
    m.run_rollup.side_effect = lambda domain, mc, subchapters, *a, **kw: {
        "domain": domain,
        "text": "Domain rollup across entities.",
        "usage": {},
    }
    m.run.side_effect = lambda domain, mc, *a, **kw: {
        "domain": domain,
        "text": "Legacy single-pass chapter text.",
        "usage": {},
    }
    return m


@pytest.mark.asyncio
async def test_synthesize_node_decomposes_entity_rich_domain_into_subchapters():
    with patch("core.graph.SynthesisAgent", return_value=_mock_synth_hierarchical()), \
         patch("core.graph.Retriever", return_value=_mock_retriever()), \
         patch("core.graph.MasterDataService", return_value=_mock_masterdata()):
        result = await synthesize_node(_entity_rich_state())

    chapters = result["synthesis_chapters"]
    competition = next(ch for ch in chapters if ch["domain"] == "competition")
    assert competition["text"] == "Domain rollup across entities."
    assert len(competition["subchapters"]) == 2
    sub_keys = {sc["subdomain_key"] for sc in competition["subchapters"]}
    assert sub_keys == {"CAT", "SAND.ST"}


@pytest.mark.asyncio
async def test_synthesize_node_kill_switch_reverts_to_legacy_single_pass():
    with patch("core.graph.SynthesisAgent", return_value=_mock_synth_hierarchical()), \
         patch("core.graph.Retriever", return_value=_mock_retriever()), \
         patch("core.graph.MasterDataService", return_value=_mock_masterdata()), \
         patch("core.graph.settings.synthesis.hierarchical_enabled", False):
        result = await synthesize_node(_entity_rich_state())

    chapters = result["synthesis_chapters"]
    competition = next(ch for ch in chapters if ch["domain"] == "competition")
    # Legacy path: SynthesisAgent.run() is used, not run_subchapter/run_rollup.
    assert competition["subchapters"] == []


@pytest.mark.asyncio
async def test_synthesize_node_does_not_interrupt_on_soft_timeout():
    """Synthesis is the final productive phase; it must not pause for the soft
    timeout (that would risk discarding finished work). interrupt() must never
    be reached even when the run is far past the soft-timeout threshold."""
    import time

    with patch("core.graph.SynthesisAgent", return_value=_mock_synth()), \
         patch("core.graph.Retriever", return_value=_mock_retriever()), \
         patch("core.graph.interrupt", side_effect=AssertionError("must not interrupt")), \
         patch("core.graph.settings.gates.gate_3_enabled", False):
        # run_start_time far in the past → elapsed >> soft_timeout_s
        result = await synthesize_node(
            _base_state(run_start_time=time.time() - 10_000, paused_seconds=0.0)
        )

    assert result["stage"] == "done"


@pytest.mark.asyncio
async def test_partial_brief_node_keeps_completed_synthesis_chapters():
    """A soft timeout that lands after synthesis finished must return the real
    chapters and executive summary, not a cruder chapter_sets rebuild."""
    chapters = [{"domain": "competition", "text": "Finished competition chapter."}]
    state = _base_state(
        synthesis_chapters=chapters,
        exec_summary="The finished executive summary.",
    )

    # SynthesisAgent must NOT be called when chapters already exist.
    synth = _mock_synth()
    with patch("core.graph.SynthesisAgent", return_value=synth):
        result = await partial_brief_node(state)

    assert result["stage"] == "done"
    assert result["synthesis_chapters"] == chapters
    assert result["exec_summary"] == "The finished executive summary."
    synth.run.assert_not_called()


@pytest.mark.asyncio
async def test_partial_brief_node_rebuilds_from_chapter_sets_when_no_synthesis():
    """With no completed synthesis_chapters, fall back to rebuilding from chapter_sets."""
    state = _base_state(synthesis_chapters=[])
    with patch("core.graph.SynthesisAgent", return_value=_mock_synth("Rebuilt text.")):
        result = await partial_brief_node(state)

    assert result["stage"] == "done"
    domains = {ch["domain"] for ch in result["synthesis_chapters"]}
    assert {"competition", "commodities"} <= domains
