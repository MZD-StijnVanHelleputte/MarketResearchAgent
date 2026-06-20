"""Tests that SynthesisAgent preserves figures through to the chapter output."""
import json
from unittest.mock import MagicMock, patch

from agents.synthesis_agent import SynthesisAgent
from core.schemas import MergedChapter


def _merged() -> MergedChapter:
    return MergedChapter(
        domain="commodities",
        text="Copper demand outlook is constructive.",
        figures={"copper_latest_USD/LB": "4.12"},
    )


def test_fallback_includes_figures_block():
    out = SynthesisAgent._fallback("commodities", _merged())
    assert out["figures"] == {"copper_latest_USD/LB": "4.12"}
    assert "**Key figures**" in out["text"]
    assert "4.12" in out["text"]


def test_run_threads_figures_through():
    """The happy (crew) path returns a figures key and appends a Key figures block."""
    crew = MagicMock()
    crew.kickoff.return_value = MagicMock(
        __str__=lambda self: json.dumps({"domain": "commodities", "text": "Polished prose."})
    )
    with patch("agents.synthesis_agent.LLM"), \
         patch("agents.synthesis_agent.Agent"), \
         patch("agents.synthesis_agent.Task"), \
         patch("agents.synthesis_agent.Crew", return_value=crew):
        agent = SynthesisAgent()
        out = agent.run("commodities", _merged(), [], [])

    assert out["figures"] == {"copper_latest_USD/LB": "4.12"}
    assert "**Key figures**" in out["text"]
    assert "4.12" in out["text"]


def test_append_figures_is_idempotent():
    text = "Body.\n\n**Key figures**\n- copper_latest_USD/LB: 4.12"
    assert SynthesisAgent._append_figures(text, {"copper_latest_USD/LB": "4.12"}) == text
