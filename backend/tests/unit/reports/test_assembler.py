import pytest
from unittest.mock import MagicMock, patch

from reports.assembler import Assembler, ReportDraft, _DOMAIN_ORDER


def _make_state(chapters: list[dict], **overrides) -> dict:
    return {
        "synthesis_chapters": chapters,
        "exec_summary": overrides.get("exec_summary", ""),
        "merge_log": overrides.get("merge_log", []),
        "warnings": overrides.get("warnings", []),
    }


def test_assemble_returns_report_draft():
    state = _make_state([{"domain": "competition", "text": "CAT is growing."}])
    draft = Assembler.assemble(state, run_id="run-001", query="CAT capex?")
    assert isinstance(draft, ReportDraft)
    assert draft.run_id == "run-001"
    assert draft.query == "CAT capex?"


def test_assemble_orders_chapters_by_domain_priority():
    chapters = [
        {"domain": "macroeconomics", "text": "Macro text."},
        {"domain": "competition", "text": "Competition text."},
        {"domain": "commodities", "text": "Commodity text."},
    ]
    state = _make_state(chapters)
    draft = Assembler.assemble(state, run_id="run-002", query="q")
    domains = [ch["domain"] for ch in draft.chapters]
    assert domains == ["commodities", "competition", "macroeconomics"]


def test_assemble_all_known_domains_in_canonical_order():
    # Build chapters in reverse canonical order
    chapters = [{"domain": d, "text": f"{d} text"} for d in reversed(_DOMAIN_ORDER)]
    state = _make_state(chapters)
    draft = Assembler.assemble(state, run_id="run-003", query="q")
    result_domains = [ch["domain"] for ch in draft.chapters]
    assert result_domains == _DOMAIN_ORDER


def test_assemble_unknown_domain_goes_last():
    chapters = [
        {"domain": "custom_domain", "text": "Unknown domain text."},
        {"domain": "competition", "text": "Competition text."},
    ]
    state = _make_state(chapters)
    draft = Assembler.assemble(state, run_id="run-004", query="q")
    domains = [ch["domain"] for ch in draft.chapters]
    assert domains[-1] == "custom_domain"
    assert domains[0] == "competition"


def test_assemble_includes_exec_summary():
    summary = "This is the executive summary with at least some content."
    state = _make_state([], exec_summary=summary)
    draft = Assembler.assemble(state, run_id="run-005", query="q")
    assert draft.exec_summary == summary


def test_assemble_empty_exec_summary_when_not_in_state():
    state = _make_state([])
    draft = Assembler.assemble(state, run_id="run-006", query="q")
    assert draft.exec_summary == ""


def test_assemble_includes_merge_log():
    state = _make_state([], merge_log=["Resolved contradiction in competition/revenue"])
    draft = Assembler.assemble(state, run_id="run-007", query="q")
    assert len(draft.merge_log) == 1
    assert "competition" in draft.merge_log[0]


def test_assemble_includes_warnings():
    state = _make_state([], warnings=["Stale chunk in commodities"])
    draft = Assembler.assemble(state, run_id="run-008", query="q")
    assert len(draft.warnings) == 1


@pytest.mark.asyncio
async def test_exec_summary_retry_on_out_of_bounds_word_count():
    """synthesize_node should retry exec summary when word count is out of bounds."""
    from unittest.mock import AsyncMock, MagicMock, patch
    from core.graph import synthesize_node, AgentState
    from core.schemas import ChapterDraft, MergedChapter

    # Build a minimal state with 1 active plan and 1 chapter_set
    plan_id = "plan_001"
    state: AgentState = {
        "run_id": "test-run",
        "session_id": "test-session",
        "user_query": "CAT capex?",
        "plans": [{
            "plan_id": plan_id,
            "domain_activations": {"competition": True, "distributors": False,
                                   "customers": False, "mining_projects": False,
                                   "commodities": False, "macro_geopolitics": False,
                                   "general_search": False},
            "tool_calls": [],
            "rationale": "test",
            "feasibility_score": 0.9,
            "quality_score": 0.85,
            "combined_score": 0.875,
            "is_survivor": True,
            "gap_report": "",
            "depth": 2,
            "estimated_token_cost": 0,
        }],
        "collection_manifest": {},
        "chapter_sets": {
            f"{plan_id}::competition": {
                "domain": "competition",
                "plan_id": plan_id,
                "text": "CAT revenue is growing.",
                "figures": {},
                "citations": [],
                "contradiction_flags": [],
            }
        },
        "synthesis_chapters": [],
        "merge_log": [],
        "confidence": 0.9,
        "react_iterations": 0,
        "context_messages": [],
        "stage": "synthesize",
        "warnings": [],
        "error": None,
        "cumulative_cost_usd": 0.0,
        "api_call_count": 0,
        "injection_flags": [],
        "clarification_done": True,
        "exec_summary": "",
    }

    # Short first response (5 words) — triggers retry
    short_response = MagicMock()
    short_response.content = "Short summary of five words."

    # Retry response with correct word count (generate a ~400+ word string)
    long_text = " ".join(["word"] * 420)
    long_response = MagicMock()
    long_response.content = long_text

    mock_llm = MagicMock()
    mock_llm.complete.side_effect = [short_response, long_response]

    mock_synth = MagicMock()
    mock_synth.run.return_value = {"domain": "competition", "text": "CAT revenue is growing."}

    with patch("core.graph.SynthesisAgent", return_value=mock_synth), \
         patch("core.graph.Retriever") as MockRetriever, \
         patch("core.graph.merge_chapter_sets") as mock_merge, \
         patch("core.graph.chapter_set_overlap", return_value=0.0), \
         patch("core.graph.LLMClient", return_value=mock_llm):

        MockRetriever.return_value.retrieve.return_value = ([], [])
        mock_merge.return_value = (
            [MergedChapter(domain="competition", text="CAT revenue is growing.")],
            [],
        )

        result = await synthesize_node(state)

    # Retry should have been called twice
    assert mock_llm.complete.call_count == 2
    # Final exec_summary should be the long (corrected) one
    assert result["exec_summary"] == long_text
    word_count = len(result["exec_summary"].split())
    assert word_count >= 400
