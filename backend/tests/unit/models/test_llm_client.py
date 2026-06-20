import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from models.llm_client import LLMClient
from models.response_parser import LLMResponse


def _make_mistral_response(tool_name: str, arguments: dict, content: str | None = None):
    """Build a minimal mock of the Mistral chat response object."""
    func = MagicMock()
    func.name = tool_name
    func.arguments = json.dumps(arguments)

    tc = MagicMock()
    tc.id = "call_abc"
    tc.function = func

    message = MagicMock()
    message.content = content
    message.tool_calls = [tc]

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 5
    usage.total_tokens = 15

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def test_complete_returns_tool_call():
    mock_response = _make_mistral_response(
        "news_search", {"query": "Caterpillar strategy"}
    )
    mock_mistral = MagicMock()
    mock_mistral.chat.complete.return_value = mock_response

    with patch("models.llm_client.settings") as mock_settings:
        mock_settings.llm.provider = "mistral"
        mock_settings.llm.model = "mistral-medium-3-5"
        mock_settings.llm.work_temperature = 0.2
        mock_settings.llm.max_tokens = 4096
        mock_settings.mistral_api_key = "test-key"

        client = LLMClient()
        client._mistral = mock_mistral

        result = client.complete(
            messages=[{"role": "user", "content": "Find Caterpillar news"}],
            tools=[{"type": "function", "function": {"name": "news_search"}}],
        )

    assert isinstance(result, LLMResponse)
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "news_search"
    assert result.tool_calls[0].arguments == {"query": "Caterpillar strategy"}
    assert result.tool_calls[0].id == "call_abc"
    assert result.usage["total_tokens"] == 15


def test_complete_parses_json_string_arguments():
    """arguments field from Mistral is a JSON string — must be parsed to dict."""
    mock_response = _make_mistral_response("news_search", {"query": "copper price"})
    mock_mistral = MagicMock()
    mock_mistral.chat.complete.return_value = mock_response

    with patch("models.llm_client.settings") as mock_settings:
        mock_settings.llm.provider = "mistral"
        mock_settings.llm.model = "mistral-medium-3-5"
        mock_settings.llm.work_temperature = 0.2
        mock_settings.llm.max_tokens = 4096
        mock_settings.mistral_api_key = "test-key"

        client = LLMClient()
        client._mistral = mock_mistral
        result = client.complete(messages=[{"role": "user", "content": "copper?"}])

    assert isinstance(result.tool_calls[0].arguments, dict)


def test_complete_text_response():
    """When Mistral returns text (no tool call), content is populated."""
    message = MagicMock()
    message.content = "This is a plain text reply."
    message.tool_calls = None

    usage = MagicMock()
    usage.prompt_tokens = 5
    usage.completion_tokens = 10
    usage.total_tokens = 15

    choice = MagicMock()
    choice.message = message

    mock_response = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = usage

    mock_mistral = MagicMock()
    mock_mistral.chat.complete.return_value = mock_response

    with patch("models.llm_client.settings") as mock_settings:
        mock_settings.llm.provider = "mistral"
        mock_settings.llm.model = "mistral-medium-3-5"
        mock_settings.llm.work_temperature = 0.2
        mock_settings.llm.max_tokens = 4096
        mock_settings.mistral_api_key = "test-key"

        client = LLMClient()
        client._mistral = mock_mistral
        result = client.complete(messages=[{"role": "user", "content": "hello"}])

    assert result.content == "This is a plain text reply."
    assert result.tool_calls == []


def test_complete_uses_max_tokens_override():
    mock_response = _make_mistral_response("news_search", {"query": "test"})
    mock_mistral = MagicMock()
    mock_mistral.chat.complete.return_value = mock_response

    with patch("models.llm_client.settings") as mock_settings:
        mock_settings.llm.provider = "mistral"
        mock_settings.llm.model = "mistral-medium-3-5"
        mock_settings.llm.work_temperature = 0.2
        mock_settings.llm.max_tokens = 4096
        mock_settings.mistral_api_key = "test-key"

        client = LLMClient()
        client._mistral = mock_mistral
        client.complete(messages=[{"role": "user", "content": "hi"}], max_tokens=8192)

    _, kwargs = mock_mistral.chat.complete.call_args
    assert kwargs["max_tokens"] == 8192


def test_complete_falls_back_to_global_max_tokens():
    mock_response = _make_mistral_response("news_search", {"query": "test"})
    mock_mistral = MagicMock()
    mock_mistral.chat.complete.return_value = mock_response

    with patch("models.llm_client.settings") as mock_settings:
        mock_settings.llm.provider = "mistral"
        mock_settings.llm.model = "mistral-medium-3-5"
        mock_settings.llm.work_temperature = 0.2
        mock_settings.llm.max_tokens = 4096
        mock_settings.mistral_api_key = "test-key"

        client = LLMClient()
        client._mistral = mock_mistral
        client.complete(messages=[{"role": "user", "content": "hi"}])

    _, kwargs = mock_mistral.chat.complete.call_args
    assert kwargs["max_tokens"] == 4096


@pytest.mark.asyncio
async def test_acomplete_uses_max_tokens_override():
    mock_response = _make_mistral_response("news_search", {"query": "test"})
    mock_mistral = MagicMock()
    mock_mistral.chat.complete_async = AsyncMock(return_value=mock_response)

    with patch("models.llm_client.settings") as mock_settings:
        mock_settings.llm.provider = "mistral"
        mock_settings.llm.model = "mistral-medium-3-5"
        mock_settings.llm.work_temperature = 0.2
        mock_settings.llm.max_tokens = 4096
        mock_settings.mistral_api_key = "test-key"

        client = LLMClient()
        client._mistral = mock_mistral
        await client.acomplete(messages=[{"role": "user", "content": "hi"}], max_tokens=8192)

    _, kwargs = mock_mistral.chat.complete_async.call_args
    assert kwargs["max_tokens"] == 8192


@pytest.mark.asyncio
async def test_acomplete_returns_llm_response():
    """acomplete uses complete_async — verify it returns the same parsed result."""
    mock_response = _make_mistral_response("news_search", {"query": "Komatsu news"})
    mock_mistral = MagicMock()
    mock_mistral.chat.complete_async = AsyncMock(return_value=mock_response)

    with patch("models.llm_client.settings") as mock_settings:
        mock_settings.llm.provider = "mistral"
        mock_settings.llm.model = "mistral-medium-3-5"
        mock_settings.llm.work_temperature = 0.2
        mock_settings.llm.max_tokens = 4096
        mock_settings.mistral_api_key = "test-key"

        client = LLMClient()
        client._mistral = mock_mistral
        result = await client.acomplete(messages=[{"role": "user", "content": "latest Komatsu news"}])

    assert isinstance(result, LLMResponse)
    assert result.tool_calls[0].name == "news_search"
    assert result.tool_calls[0].arguments == {"query": "Komatsu news"}


@pytest.mark.asyncio
async def test_acomplete_text_response():
    """acomplete propagates plain-text content (no tool call) correctly."""
    message = MagicMock()
    message.content = "Here is your summary."
    message.tool_calls = None

    usage = MagicMock()
    usage.prompt_tokens = 8
    usage.completion_tokens = 12
    usage.total_tokens = 20

    choice = MagicMock()
    choice.message = message

    mock_response = MagicMock()
    mock_response.choices = [choice]
    mock_response.usage = usage

    mock_mistral = MagicMock()
    mock_mistral.chat.complete_async = AsyncMock(return_value=mock_response)

    with patch("models.llm_client.settings") as mock_settings:
        mock_settings.llm.provider = "mistral"
        mock_settings.llm.model = "mistral-medium-3-5"
        mock_settings.llm.work_temperature = 0.2
        mock_settings.llm.max_tokens = 4096
        mock_settings.mistral_api_key = "test-key"

        client = LLMClient()
        client._mistral = mock_mistral
        result = await client.acomplete(messages=[{"role": "user", "content": "summarise"}])

    assert result.content == "Here is your summary."


def _make_embedding_response(n: int):
    response = MagicMock()
    response.data = [MagicMock(embedding=[float(i)]) for i in range(n)]
    return response


def test_embed_batch_sends_one_request_for_small_input():
    """A batch within batch_size should hit the API exactly once, with all texts."""
    mock_mistral = MagicMock()
    mock_mistral.embeddings.create.return_value = _make_embedding_response(3)

    client = LLMClient()
    client._mistral = mock_mistral

    result = client.embed_batch(["a", "b", "c"], batch_size=64)

    assert mock_mistral.embeddings.create.call_count == 1
    _, kwargs = mock_mistral.embeddings.create.call_args
    assert kwargs["inputs"] == ["a", "b", "c"]
    assert len(result) == 3


def test_embed_batch_splits_into_multiple_requests():
    """More texts than batch_size should split across multiple API calls."""
    mock_mistral = MagicMock()
    mock_mistral.embeddings.create.side_effect = [
        _make_embedding_response(2),
        _make_embedding_response(1),
    ]

    client = LLMClient()
    client._mistral = mock_mistral

    result = client.embed_batch(["a", "b", "c"], batch_size=2)

    assert mock_mistral.embeddings.create.call_count == 2
    assert len(result) == 3


def test_embed_batch_splits_on_cumulative_byte_budget():
    """Dense chunks near the per-chunk byte cap must not be packed 64-at-a-time —
    their summed bytes can exceed Mistral's per-request token budget even though
    each individual chunk and the item count are within limits."""
    from models import llm_client as llm_client_module

    big_text = "x" * 7700  # near Chunker._MAX_BYTES / _MAX_EMBED_BYTES (7800)
    texts = [big_text] * 30  # 30 * 7700 = 231,000 bytes, far over _MAX_BATCH_BYTES

    mock_mistral = MagicMock()
    mock_mistral.embeddings.create.side_effect = lambda model, inputs: _make_embedding_response(
        len(inputs)
    )

    client = LLMClient()
    client._mistral = mock_mistral

    progress_calls = []
    result = client.embed_batch(
        texts, batch_size=64, on_progress=lambda done, total: progress_calls.append((done, total))
    )

    assert len(result) == 30
    assert mock_mistral.embeddings.create.call_count > 1
    for _, kwargs in mock_mistral.embeddings.create.call_args_list:
        batch = kwargs["inputs"]
        assert sum(len(t.encode("utf-8")) for t in batch) <= llm_client_module._MAX_BATCH_BYTES
        assert len(batch) <= 64
    assert progress_calls[-1] == (30, 30)


def test_embed_batch_respects_both_count_and_byte_limits():
    """Small texts well under the byte budget should still respect batch_size."""
    mock_mistral = MagicMock()
    mock_mistral.embeddings.create.side_effect = lambda model, inputs: _make_embedding_response(
        len(inputs)
    )

    client = LLMClient()
    client._mistral = mock_mistral

    texts = ["short"] * 10
    result = client.embed_batch(texts, batch_size=3)

    assert len(result) == 10
    assert mock_mistral.embeddings.create.call_count == 4  # ceil(10/3)
    for _, kwargs in mock_mistral.embeddings.create.call_args_list[:-1]:
        assert len(kwargs["inputs"]) == 3
