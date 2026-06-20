import json
from typing import Callable

from config import settings
from models.response_parser import LLMResponse, ToolCall


class LLMClient:
    """Wrapper over the Mistral chat/embeddings API."""

    def __init__(self) -> None:
        self._model = settings.llm.model
        self._mistral = None  # lazily instantiated

    def _get_mistral(self):
        if self._mistral is None:
            from mistralai.client import Mistral
            self._mistral = Mistral(
                api_key=settings.mistral_api_key,
                timeout_ms=settings.llm.llm_timeout_s * 1000,  # SDK wants milliseconds
            )
        return self._mistral

    def _parse_response(self, response) -> LLMResponse:
        message = response.choices[0].message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                raw_args = tc.function.arguments
                arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))

        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=message.content if not tool_calls else None,
            tool_calls=tool_calls,
            usage=usage,
        )

    def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else settings.llm.work_temperature
        tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens
        return self._complete_mistral(messages, tools, temp, tokens)

    def _complete_mistral(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        client = self._get_mistral()
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = client.chat.complete(**kwargs)
        return self._parse_response(response)

    async def acomplete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        temp = temperature if temperature is not None else settings.llm.work_temperature
        tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens
        return await self._acomplete_mistral(messages, tools, temp, tokens)

    async def _acomplete_mistral(
        self,
        messages: list[dict],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._get_mistral().chat.complete_async(**kwargs)
        return self._parse_response(response)

    def embed(self, text: str) -> list[float]:
        client = self._get_mistral()
        result = client.embeddings.create(model="mistral-embed", inputs=text)
        return result.data[0].embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Embeds many texts in as few API round-trips as possible.

        Mistral's embeddings endpoint accepts a list of inputs per request, so this
        sends `batch_size` texts per call instead of one call per text. If given,
        `on_progress(embedded_so_far, total)` is called after each batch lands.
        """
        client = self._get_mistral()
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = client.embeddings.create(model="mistral-embed", inputs=batch)
            embeddings.extend(d.embedding for d in result.data)
            if on_progress:
                on_progress(len(embeddings), len(texts))
        return embeddings
