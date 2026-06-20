"""Bounded ReAct research loop that resolves entities and discovers market context
before the Tree-of-Thought planner generates candidate plans.

The agent may call web_search, news_search, web_extract, and masterdata_lookup
up to max_calls times, then returns a ResearchContext summarising what it found.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time

from config import settings
from core.tot.schemas import ResearchContext
from models.llm_client import LLMClient
from prompts.research_prompt import research_messages
from prompts.tool_schemas import RESEARCH_SCHEMAS
from core.tool_router import async_route

logger = logging.getLogger(__name__)

_FALLBACK_CONTEXT = ResearchContext()


def _parse_research_json(raw: str) -> dict:
    """Extract JSON from the LLM's research-findings response."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    # Find the outermost {...}
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in research response")
    return json.loads(text[start:end])


class ResearchAgent:
    """
    Performs a short, bounded ReAct loop to resolve entities and gather
    market context before the ToT planner generates plans.

    Args:
        max_calls: Hard cap on tool calls (default from settings).
        timeout_s:  Wall-clock timeout for the entire research loop.
    """

    def __init__(
        self,
        max_calls: int | None = None,
        timeout_s: int | None = None,
    ) -> None:
        self._max_calls = max_calls or settings.understand.research_max_tool_calls
        self._timeout_s = timeout_s or settings.understand.research_timeout_s
        self._llm = LLMClient()
        self.last_usage: dict = {}

    async def run(self, query: str) -> ResearchContext:
        """Run the research loop for the given query.

        Returns a ResearchContext. Never raises — on any failure returns an
        empty ResearchContext so the caller can proceed without research.
        """
        try:
            return await asyncio.wait_for(
                self._loop(query),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("ResearchAgent: timed out after %ss — using empty context", self._timeout_s)
            return _FALLBACK_CONTEXT
        except Exception as exc:
            logger.warning("ResearchAgent: failed (%s) — using empty context", exc)
            return _FALLBACK_CONTEXT

    async def _loop(self, query: str) -> ResearchContext:
        messages = research_messages(query, self._max_calls)
        calls_made = 0
        total_usage: dict = {}

        while calls_made < self._max_calls:
            response = await self._llm.acomplete(
                messages,
                tools=RESEARCH_SCHEMAS,
                temperature=settings.llm.work_temperature,
            )
            _merge_usage(total_usage, response.usage)

            if not response.tool_calls:
                # LLM returned a final text answer — parse it
                self.last_usage = total_usage
                return self._parse_context(response.content or "", calls_made)

            # Execute every tool call the LLM requested (may be batched)
            tool_results_msg: dict = {"role": "tool", "content": []}
            for tc in response.tool_calls:
                if calls_made >= self._max_calls:
                    break
                try:
                    result = await async_route(tc.name, tc.arguments)
                    result_text = json.dumps(result)
                except Exception as exc:
                    result_text = json.dumps({"error": str(exc)})
                calls_made += 1
                tool_results_msg["content"].append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": result_text,
                })

            # Append assistant message + tool results to the conversation
            messages = messages + [
                {"role": "assistant", "content": None, "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]},
                tool_results_msg,
            ]

        # Max calls reached — ask LLM to summarise what it has
        messages.append({
            "role": "user",
            "content": (
                "You have used the maximum number of tool calls. "
                "Summarise your findings as the required JSON object now."
            ),
        })
        final = await self._llm.acomplete(messages, temperature=settings.llm.work_temperature)
        _merge_usage(total_usage, final.usage)
        self.last_usage = total_usage
        return self._parse_context(final.content or "", calls_made)

    def _parse_context(self, raw: str, calls_used: int) -> ResearchContext:
        try:
            data = _parse_research_json(raw)
            data["tool_calls_used"] = calls_used
            return ResearchContext(**{k: v for k, v in data.items() if k in ResearchContext.model_fields})
        except Exception as exc:
            logger.warning("ResearchAgent: could not parse JSON output (%s)", exc)
            return ResearchContext(tool_calls_used=calls_used)


def _merge_usage(acc: dict, delta: dict) -> None:
    for k, v in delta.items():
        acc[k] = acc.get(k, 0) + v
