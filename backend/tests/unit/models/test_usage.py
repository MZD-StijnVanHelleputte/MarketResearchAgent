import pytest

from models import usage as usage_mod


def test_usd_cost_uses_configured_prices(monkeypatch):
    monkeypatch.setattr("models.usage.settings.llm.input_price_per_1m", 0.40)
    monkeypatch.setattr("models.usage.settings.llm.output_price_per_1m", 2.00)
    # 1M input @ $0.40 + 0.5M output @ $2.00 = 0.40 + 1.00
    assert usage_mod.usd_cost(1_000_000, 500_000) == pytest.approx(1.40)


def test_usd_cost_zero_tokens():
    assert usage_mod.usd_cost(0, 0) == 0.0


def test_llm_usage_normalises_and_counts_one_request():
    u = usage_mod.llm_usage({"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    assert u == {"prompt_tokens": 10, "completion_tokens": 5, "requests": 1}


def test_llm_usage_empty_is_zero_requests():
    assert usage_mod.llm_usage(None) == {"prompt_tokens": 0, "completion_tokens": 0, "requests": 0}


def test_crew_usage_reads_token_usage_attr():
    class _TU:
        prompt_tokens = 100
        completion_tokens = 40
        successful_requests = 2

    class _Out:
        token_usage = _TU()

    assert usage_mod.crew_usage(_Out()) == {
        "prompt_tokens": 100, "completion_tokens": 40, "requests": 2,
    }


def test_crew_usage_missing_attr_is_zero():
    assert usage_mod.crew_usage(object()) == {
        "prompt_tokens": 0, "completion_tokens": 0, "requests": 0,
    }


def test_accumulate_adds_to_prior_state_totals(monkeypatch):
    monkeypatch.setattr("models.usage.settings.llm.input_price_per_1m", 1.00)
    monkeypatch.setattr("models.usage.settings.llm.output_price_per_1m", 1.00)
    state = {"prompt_tokens": 100, "completion_tokens": 100, "api_call_count": 1}
    delta = usage_mod.accumulate(
        state,
        {"prompt_tokens": 50, "completion_tokens": 25, "requests": 2},
    )
    assert delta["prompt_tokens"] == 150
    assert delta["completion_tokens"] == 125
    assert delta["total_tokens"] == 275
    assert delta["api_call_count"] == 3
    assert delta["cumulative_cost_usd"] == pytest.approx(275 / 1_000_000)
