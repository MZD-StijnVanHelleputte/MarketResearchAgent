"""Unit tests for the run-scoped tool circuit breaker and its integration with
async_route — tool-level failure tracking that blocks a tool after N failures (or
immediately on a rate limit) so it can't be re-hammered or pollute downstream."""
import pytest

import tools.registry as registry
from clients.base_http_client import ClientError
from config import settings
from core import tool_circuit_breaker as breaker
from core.tool_circuit_breaker import ToolBlockedError, is_rate_limited_error
from core.event_logger import set_run_context
from core.tool_router import async_route


@pytest.fixture(autouse=True)
def _clean_breaker():
    # Each test gets a unique run id; clear afterward so module-level state can't leak.
    yield
    breaker._runs.clear()


# ── Classifier ───────────────────────────────────────────────────────────────

def test_is_rate_limited_error_on_429():
    assert is_rate_limited_error(ClientError(429, "Too Many Requests"))


def test_is_rate_limited_error_by_message():
    assert is_rate_limited_error(Exception("You have hit the rate limit, slow down"))


def test_is_rate_limited_error_walks_cause_chain():
    inner = ClientError(429, "throttled")
    try:
        try:
            raise inner
        except ClientError as exc:
            raise RuntimeError("tool wrapper failed") from exc
    except RuntimeError as wrapped:
        assert is_rate_limited_error(wrapped)


def test_permanent_400_is_not_rate_limited():
    assert not is_rate_limited_error(ClientError(400, "bad request"))
    assert not is_rate_limited_error(ClientError(404, "not found"))


# ── Failure accounting ───────────────────────────────────────────────────────

def test_blocks_after_threshold_failures():
    rid = "run-threshold"
    breaker.reset_run(rid)
    n = settings.react.tool_failure_threshold
    for i in range(n - 1):
        just_blocked = breaker.record_failure(rid, "get_company_financials", ValueError("boom"))
        assert not just_blocked
        assert not breaker.is_blocked(rid, "get_company_financials")
    # The Nth failure trips the breaker.
    assert breaker.record_failure(rid, "get_company_financials", ValueError("boom"))
    assert breaker.is_blocked(rid, "get_company_financials")


def test_rate_limit_blocks_immediately():
    rid = "run-429"
    breaker.reset_run(rid)
    assert breaker.record_failure(rid, "get_equity_price", ClientError(429, "Too Many Requests"))
    assert breaker.is_blocked(rid, "get_equity_price")
    assert "rate" in breaker.block_reason(rid, "get_equity_price").lower()


def test_block_is_per_tool_not_global():
    rid = "run-isolation"
    breaker.reset_run(rid)
    breaker.record_failure(rid, "tool_a", ClientError(429, "rl"))
    assert breaker.is_blocked(rid, "tool_a")
    assert not breaker.is_blocked(rid, "tool_b")


def test_success_does_not_unblock():
    rid = "run-sticky"
    breaker.reset_run(rid)
    breaker.record_failure(rid, "tool_a", ClientError(429, "rl"))
    breaker.record_success(rid, "tool_a")  # sticky for the rest of the run
    assert breaker.is_blocked(rid, "tool_a")


def test_reset_run_clears_state():
    rid = "run-reset"
    breaker.record_failure(rid, "tool_a", ClientError(429, "rl"))
    assert breaker.is_blocked(rid, "tool_a")
    breaker.reset_run(rid)
    assert not breaker.is_blocked(rid, "tool_a")


def test_summary_reports_blocked_tools():
    rid = "run-summary"
    breaker.reset_run(rid)
    breaker.record_failure(rid, "tool_a", ClientError(429, "rl"))
    breaker.record_failure(rid, "tool_b", ValueError("x"))
    summ = breaker.summary(rid)
    assert summ["tool_a"]["blocked"] is True
    assert summ["tool_b"]["blocked"] is False
    assert summ["tool_b"]["fail_count"] == 1


def test_no_run_id_is_safe():
    assert not breaker.is_blocked(None, "tool_a")
    assert not breaker.record_failure(None, "tool_a", ValueError("x"))
    assert breaker.summary(None) == {}


# ── async_route integration ──────────────────────────────────────────────────

class _FakeTool:
    def __init__(self, *, fail=False):
        self.calls = 0
        self._fail = fail

    async def run(self, **kwargs):
        self.calls += 1
        if self._fail:
            raise ValueError("boom")
        return {"ok": 1}


@pytest.fixture(autouse=True)
def _silence_router_io(monkeypatch):
    # async_route logs events and records live sources via SqliteStore; stub them so
    # these pure unit tests don't touch the database.
    async def _noop(*args, **kwargs):
        return None
    monkeypatch.setattr("core.tool_router.log_event", _noop)
    monkeypatch.setattr("core.tool_router.record_live_source", _noop)


async def test_async_route_skips_blocked_tool_without_calling_it(monkeypatch):
    rid = "run-skip"
    breaker.reset_run(rid)
    for _ in range(settings.react.tool_failure_threshold):
        breaker.record_failure(rid, "get_company_financials", ValueError("boom"))
    assert breaker.is_blocked(rid, "get_company_financials")

    fake = _FakeTool()
    monkeypatch.setattr(registry, "get", lambda name: fake)
    set_run_context(rid)

    with pytest.raises(ToolBlockedError):
        await async_route("get_company_financials", {})
    assert fake.calls == 0  # the real tool was never invoked


async def test_async_route_count_failures_flag(monkeypatch):
    rid = "run-count"
    breaker.reset_run(rid)
    fake = _FakeTool(fail=True)
    monkeypatch.setattr(registry, "get", lambda name: fake)
    set_run_context(rid)

    # count_failures=False (the domain-agent repair path) must NOT inflate the count.
    with pytest.raises(ValueError):
        await async_route("toolX", {}, count_failures=False)
    assert breaker.summary(rid) == {}

    # count_failures=True (direct callers) records exactly one failure.
    with pytest.raises(ValueError):
        await async_route("toolX", {}, count_failures=True)
    assert breaker.summary(rid)["toolX"]["fail_count"] == 1


async def test_async_route_success_records_success(monkeypatch):
    rid = "run-ok"
    breaker.reset_run(rid)
    fake = _FakeTool()
    monkeypatch.setattr(registry, "get", lambda name: fake)
    set_run_context(rid)

    result = await async_route("toolX", {"a": 1})
    assert result == {"ok": 1}
    assert fake.calls == 1
    assert not breaker.is_blocked(rid, "toolX")
