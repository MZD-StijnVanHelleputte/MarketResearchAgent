"""Regression tests: a malformed LLM JSON envelope from a domain agent must never
leak raw JSON into chapter text. Either the prose is recovered, or parsing raises so
run()'s existing fallback (which preserves figures/citations from raw tool results) is used."""
import pytest

from agents.commodities_agent import CommoditiesAgent


def test_parse_result_recovers_text_when_sibling_key_is_malformed():
    agent = CommoditiesAgent()
    raw = (
        '{"domain": "commodities", "text": "Copper at $4.12/lb.", '
        '"citations": ["unterminated'
    )
    draft = agent._parse_result(raw, "plan-1")
    assert draft.text == "Copper at $4.12/lb."
    assert "{" not in draft.text


def test_parse_result_raises_on_totally_unparseable_response():
    agent = CommoditiesAgent()
    with pytest.raises(ValueError):
        agent._parse_result("completely unparseable, no json here", "plan-1")
