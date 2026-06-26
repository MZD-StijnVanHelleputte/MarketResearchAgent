"""End-to-end tests: tool-calling slice (Phase 1) and full chat run (Phase 2.6).

Requires MISTRAL_API_KEY and NEWSAPI_API_KEY set in backend/.env (or environment).
Run with: pytest tests/integration/test_tool_call_e2e.py -v -m integration
"""
import pytest
from models.llm_client import LLMClient
from prompts.tool_schemas import TOOL_SCHEMAS
from tools.registry import get as get_tool


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_calls_news_search_tool():
    client = LLMClient()

    response = client.complete(
        messages=[
            {
                "role": "user",
                "content": (
                    "Search for the latest news about Caterpillar construction equipment strategy."
                ),
            }
        ],
        tools=TOOL_SCHEMAS,
    )

    assert len(response.tool_calls) > 0, "Expected at least one tool call from the LLM"
    tc = response.tool_calls[0]
    assert tc.name == "news_search", f"Expected 'news_search', got '{tc.name}'"
    assert "query" in tc.arguments, "Tool call must include a 'query' argument"
    assert isinstance(tc.arguments["query"], str)
    assert len(tc.arguments["query"]) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_tool_executes_and_returns_articles():
    """Full round-trip: LLM selects tool → tool runs → articles returned."""
    client = LLMClient()

    response = client.complete(
        messages=[
            {
                "role": "user",
                "content": "Find recent news about Komatsu autonomous mining trucks.",
            }
        ],
        tools=TOOL_SCHEMAS,
    )

    assert response.tool_calls, "LLM must call a tool"
    tc = response.tool_calls[0]
    tool = get_tool(tc.name)
    result = await tool.run(**tc.arguments)

    assert "articles" in result
    assert isinstance(result["articles"], list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_post_chat_completes():
    """Phase 2.6 gate: POST /chat runs the full graph and returns a non-empty brief."""
    import asyncio
    import time

    from httpx import AsyncClient, ASGITransport
    from api.main import app

    # Seed required entity preferences so the clarification gate does not fire.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        await ac.put("/preferences", json={
            "equipment_models": ["PC7000", "730E"],
            "operators": ["Rio Tinto", "BHP"],
            "competitor_tickers": ["CAT", "VOLV-B.ST"],
        })

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post("/chat", json={"query": "What is the current copper price?"})
    assert resp.status_code == 200
    data = resp.json()
    run_id = data["run_id"]
    assert data["status"] == "running"

    deadline = time.monotonic() + 180
    row: dict = {}
    while time.monotonic() < deadline:
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            poll = await ac.get(f"/runs/{run_id}")
        assert poll.status_code == 200
        row = poll.json()
        if row["status"] in ("done", "error"):
            break
        await asyncio.sleep(5)

    assert row["status"] == "done", (
        f"Run ended with status={row['status']}, error={row.get('error')}"
    )
    assert row.get("brief"), "Brief must be non-empty"
