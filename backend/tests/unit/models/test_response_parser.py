from models.response_parser import LLMResponse, ToolCall


def test_tool_call_fields():
    tc = ToolCall(id="call_1", name="news_search", arguments={"query": "Caterpillar"})
    assert tc.id == "call_1"
    assert tc.name == "news_search"
    assert tc.arguments == {"query": "Caterpillar"}


def test_llm_response_defaults():
    resp = LLMResponse(content="hello")
    assert resp.content == "hello"
    assert resp.tool_calls == []
    assert resp.usage == {}


def test_llm_response_with_tool_calls():
    tc = ToolCall(id="c1", name="news_search", arguments={"query": "copper"})
    resp = LLMResponse(content=None, tool_calls=[tc], usage={"total_tokens": 42})
    assert resp.content is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "news_search"
    assert resp.usage["total_tokens"] == 42
