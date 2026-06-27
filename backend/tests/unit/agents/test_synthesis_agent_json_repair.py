"""Regression tests: a malformed LLM JSON envelope must never leak raw JSON into
chapter text — it should fall back to the agent's existing placeholder text instead."""
from unittest.mock import MagicMock, patch

from agents.synthesis_agent import SynthesisAgent
from core.schemas import MergedChapter


def _merged() -> MergedChapter:
    return MergedChapter(domain="customers", text="Fallback chapter text.")


def _crew_returning(raw: str) -> MagicMock:
    crew = MagicMock()
    crew.kickoff.return_value = MagicMock(__str__=lambda self: raw)
    return crew


def test_run_falls_back_on_unparseable_json_instead_of_leaking_it():
    broken = '{ "domain": "customers", "text": '  # truncated, unrecoverable
    with patch("agents.synthesis_agent.LLM"), \
         patch("agents.synthesis_agent.Agent"), \
         patch("agents.synthesis_agent.Task"), \
         patch("agents.synthesis_agent.Crew", return_value=_crew_returning(broken)):
        agent = SynthesisAgent()
        out = agent.run("customers", _merged(), [], [])
    assert "{" not in out["text"]
    assert out["text"] == "Fallback chapter text."


def test_run_rollup_falls_back_on_unparseable_json():
    broken = "not json at all"
    with patch("agents.synthesis_agent.LLM"), \
         patch("agents.synthesis_agent.Agent"), \
         patch("agents.synthesis_agent.Task"), \
         patch("agents.synthesis_agent.Crew", return_value=_crew_returning(broken)):
        agent = SynthesisAgent()
        out = agent.run_rollup("customers", _merged(), [{"subdomain_label": "BHP", "text": "BHP detail."}])
    assert "{" not in out["text"]
    assert out["synthesis_error"] is not None


def test_run_recovers_text_when_only_sibling_key_is_malformed():
    raw = '{"domain": "customers", "text": "Polished prose.", "extra": "unterminated'
    with patch("agents.synthesis_agent.LLM"), \
         patch("agents.synthesis_agent.Agent"), \
         patch("agents.synthesis_agent.Task"), \
         patch("agents.synthesis_agent.Crew", return_value=_crew_returning(raw)):
        agent = SynthesisAgent()
        out = agent.run("customers", _merged(), [], [])
    assert out["text"].startswith("Polished prose.")
