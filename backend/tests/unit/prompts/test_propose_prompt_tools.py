"""Guard against tool-name drift between the planner prompt and the registry.

The ToT proposer teaches the LLM which tools exist. If those names diverge from
the registered tool `.name` values, grounding drops the calls and async_route
raises KeyError — silently degrading every plan. These tests pin the contract.
"""
import json
import re

from prompts import propose_prompt
from prompts.propose_prompt import propose_messages
from tools import registry
from tools.registry import COLLECT_TOOLS


_COLLECT_NAMES = {t.name for t in COLLECT_TOOLS}


def test_taught_tool_names_are_all_registered():
    """Every name in the prompt's _TOOLS list must resolve via registry.get()."""
    for name in propose_prompt._TOOLS:
        registry.get(name)  # raises KeyError if unknown


def test_taught_tool_names_are_all_collect_stage_tools():
    """Planner collect-stage tools must be actual COLLECT_TOOLS, not RAG tools."""
    assert set(propose_prompt._TOOLS) <= _COLLECT_NAMES


def test_system_prompt_lists_only_registered_tools():
    """The 'Available tools:' line in the system prompt must not name unknowns."""
    messages = propose_messages("test query", [], n=3, min_dims=3)
    system = messages[0]["content"]
    line = next(l for l in system.splitlines() if l.startswith("Available tools:"))
    named = {t.strip() for t in line.removeprefix("Available tools:").split(",")}
    unknown = {n for n in named if n} - _COLLECT_NAMES
    assert not unknown, f"prompt names unregistered tools: {unknown}"


def test_worked_example_tool_calls_are_routable():
    """The JSON example embedded in the system prompt must use real tool names."""
    system = propose_prompt._SYSTEM
    for tool_name in re.findall(r'"tool":\s*"([^"]+)"', system):
        registry.get(tool_name)  # raises KeyError if the example drifts
