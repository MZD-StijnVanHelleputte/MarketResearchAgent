import json
import logging
from typing import Callable

from config import settings
from models.response_parser import LLMResponse, ToolCall

logger = logging.getLogger(__name__)

# Conservative UTF-8 byte ceiling for a single embedding input, mirroring
# Chunker._MAX_BYTES.  Belt-and-suspenders guard in case an input reaches
# this client without going through Chunker's own length cap.  Byte length
# is a hard upper bound on token count for byte-level BPE tokenizers, unlike
# estimating with a different vendor's tokenizer (which proved unreliable).
_MAX_EMBED_BYTES = 7800

# Mistral's mistral-embed batch endpoint caps total tokens per request at
# 16384 (and item count at 128). Bytes upper-bound tokens for byte-level
# BPE tokenizers (see _MAX_EMBED_BYTES), so capping cumulative batch bytes
# below 16384 guarantees the token budget is never exceeded either.
_MAX_BATCH_BYTES = 16_000


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

    @staticmethod
    def _guard_embed_length(text: str) -> str:
        data = text.encode("utf-8")
        if len(data) <= _MAX_EMBED_BYTES:
            return text
        logger.warning(
            "Embedding input of %d bytes exceeds safety cap; truncating to %d bytes.",
            len(data), _MAX_EMBED_BYTES,
        )
        return data[:_MAX_EMBED_BYTES].decode("utf-8", errors="ignore")

    def embed(self, text: str) -> list[float]:
        client = self._get_mistral()
        result = client.embeddings.create(
            model="mistral-embed", inputs=self._guard_embed_length(text)
        )
        return result.data[0].embedding

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int = 64,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> list[list[float]]:
        """Embeds many texts in as few API round-trips as possible.

        Mistral's embeddings endpoint accepts a list of inputs per request, so this
        sends `batch_size` texts per call instead of one call per text. Batches are
        also split whenever their cumulative byte length would exceed
        `_MAX_BATCH_BYTES`, since Mistral enforces a total-token budget per request
        in addition to the item-count limit — a batch of `batch_size` chunks that
        are each individually within `_MAX_EMBED_BYTES` can still collectively
        exceed that budget. If given, `on_progress(embedded_so_far, total)` is
        called after each batch lands.
        """
        client = self._get_mistral()
        guarded = [self._guard_embed_length(t) for t in texts]

        batches: list[list[str]] = []
        current: list[str] = []
        current_bytes = 0
        for text in guarded:
            text_bytes = len(text.encode("utf-8"))
            if current and (
                len(current) >= batch_size or current_bytes + text_bytes > _MAX_BATCH_BYTES
            ):
                batches.append(current)
                current, current_bytes = [], 0
            current.append(text)
            current_bytes += text_bytes
        if current:
            batches.append(current)

        embeddings: list[list[float]] = []
        for batch in batches:
            result = client.embeddings.create(model="mistral-embed", inputs=batch)
            embeddings.extend(d.embedding for d in result.data)
            if on_progress:
                on_progress(len(embeddings), len(texts))
        return embeddings
