import pytest

from memory.context_window import ContextWindow


def test_add_under_budget_keeps_all_messages():
    cw = ContextWindow(budget=1000)
    cw.add("user", "Hello")
    cw.add("assistant", "World")
    assert len(cw.to_state()) == 2


def test_to_prompt_messages_omits_token_count():
    cw = ContextWindow(budget=1000)
    cw.add("user", "Hello")
    msgs = cw.to_prompt_messages()
    assert msgs == [{"role": "user", "content": "Hello"}]
    assert "token_count" not in msgs[0]


def test_as_messages_is_alias():
    cw = ContextWindow(budget=1000)
    cw.add("user", "Hello")
    assert cw.as_messages() == cw.to_prompt_messages()


def test_evicts_oldest_when_over_budget():
    # Each message is 40 chars → 10 tokens. Budget = 15 → allows 1 message.
    cw = ContextWindow(budget=15)
    msg = "A" * 40  # 10 tokens
    cw.add("user", msg)
    cw.add("assistant", msg)
    cw.add("user", msg)
    # After three 10-token adds: total=30 > 15, prune down.
    # Should keep only 1 message (the last one added).
    state = cw.to_state()
    assert len(state) == 1
    assert state[0]["role"] == "user"


def test_single_message_never_evicted_even_if_over_budget():
    # Even if one message exceeds budget, it stays (len > 1 guard).
    cw = ContextWindow(budget=1)
    cw.add("user", "A" * 400)  # 100 tokens >> budget of 1
    assert len(cw.to_state()) == 1


def test_from_state_to_state_round_trip():
    original = [
        {"role": "user", "content": "Hi", "token_count": 1},
        {"role": "assistant", "content": "Hello", "token_count": 1},
    ]
    cw = ContextWindow.from_state(original, budget=1000)
    assert cw.to_state() == original


def test_from_state_empty():
    cw = ContextWindow.from_state([], budget=1000)
    assert cw.to_prompt_messages() == []
