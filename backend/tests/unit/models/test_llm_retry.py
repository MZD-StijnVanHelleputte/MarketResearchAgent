"""Unit tests for models/llm_retry.py (CrewAI/Mistral retry-with-backoff)."""
import pytest

from models.llm_retry import call_with_backoff, is_retryable


def test_is_retryable_matches_rate_limit_markers():
    assert is_retryable(Exception("HTTP 429: Too Many Requests"))
    assert is_retryable(Exception("Connection timed out"))
    assert not is_retryable(Exception("invalid JSON in prompt"))


def test_call_with_backoff_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("models.llm_retry.time.sleep", lambda *_: None)
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise Exception("429 rate limit exceeded")
        return "ok"

    result = call_with_backoff(flaky, max_attempts=3, base_delay=0.01)
    assert result == "ok"
    assert attempts["n"] == 3


def test_call_with_backoff_raises_immediately_on_non_retryable():
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("malformed prompt")

    with pytest.raises(ValueError):
        call_with_backoff(broken, max_attempts=3, base_delay=0.01)
    assert calls["n"] == 1


def test_call_with_backoff_raises_after_exhausting_attempts(monkeypatch):
    monkeypatch.setattr("models.llm_retry.time.sleep", lambda *_: None)

    def always_fails():
        raise Exception("503 service unavailable")

    with pytest.raises(Exception, match="503"):
        call_with_backoff(always_fails, max_attempts=2, base_delay=0.01)
