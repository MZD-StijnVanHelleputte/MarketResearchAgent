"""Unit test for ResearchAgent's tool-result message shape.

core/research_agent.py used to build tool results in Anthropic's nested
content-block shape ({"role": "tool", "content": [{"type": "tool_result",
"tool_use_id": ...}, ...]}), but this project only talks to Mistral, whose
API expects one flat {"role": "tool", "tool_call_id": ..., "content": "<str>"}
message per tool call. The old shape failed Mistral's schema validation
(union_tag_invalid) and silently degraded every run to an empty research
context.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.research_agent import ResearchAgent
from models.response_parser import LLMResponse, ToolCall


@pytest.mark.asyncio
async def test_tool_results_use_flat_mistral_shape(monkeypatch):
    agent = ResearchAgent(max_calls=2, timeout_s=5)

    tool_call = ToolCall(id="call_1", name="masterdata_lookup", arguments={"entity_type": "competitors"})
    first_response = LLMResponse(content=None, tool_calls=[tool_call], usage={})
    final_response = LLMResponse(content='{"summary": "ok"}', tool_calls=[], usage={})

    agent._llm.acomplete = AsyncMock(side_effect=[first_response, final_response])

    monkeypatch.setattr(
        "core.research_agent.async_route",
        AsyncMock(return_value={"results": []}),
    )

    await agent._loop("copper demand")

    # Inspect the messages passed into the second acomplete call.
    second_call_messages = agent._llm.acomplete.call_args_list[1].args[0]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]

    assert len(tool_messages) == 1
    msg = tool_messages[0]
    assert msg["tool_call_id"] == "call_1"
    assert msg["name"] == "masterdata_lookup"
    assert isinstance(msg["content"], str)
    assert "type" not in msg
    assert "tool_use_id" not in msg
