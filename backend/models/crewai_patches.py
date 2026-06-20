"""Workaround for a crewai 1.14.6 bug affecting Mistral.

`crewai.agents.crew_agent_executor._setup_messages()` unconditionally marks the
first system/user message of every turn with `cache_breakpoint: True` (an
Anthropic-style prompt-caching hint, via `crewai.llms.cache.mark_cache_breakpoint`).
That flag is stripped before hitting the wire for crewai's native provider
adapters (Anthropic/OpenAI/Azure/Bedrock/Gemini), but Mistral has no native
adapter — it goes through the generic litellm fallback, whose
`_format_messages_for_provider()` never strips the flag. Mistral's API rejects
the unknown field with `extra_forbidden`.

This project only ever talks to Mistral (`config.settings.LLMSettings.provider`
is pinned to `"mistral"`), so disabling the marker entirely costs nothing — it
was never going to produce a caching benefit against Mistral's API anyway.

Delete this module once crewai ships a Mistral-aware strip, or revisit it if
this project ever adds a non-Mistral provider.
"""
import crewai.llms.cache as _cache


def _noop_mark_cache_breakpoint(message: dict) -> dict:
    return message


_cache.mark_cache_breakpoint = _noop_mark_cache_breakpoint
