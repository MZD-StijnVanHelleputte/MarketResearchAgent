"""Retry/backoff and global concurrency control for CrewAI/Mistral calls.

CrewAI's `crew.kickoff()` has no built-in retry, so a single transient 429 or
timeout permanently loses whatever evidence was passed into that call (the
caller falls back to a generic placeholder). This module gives synthesis_agent
and base_domain_agent a shared way to ride out transient errors instead of
giving up on the first one, and a global cap on concurrent calls so a run
with many domains/entities doesn't fire dozens of simultaneous requests at
the same API key.
"""
import logging
import threading
import time

from config import settings

logger = logging.getLogger(__name__)

# Per-process cap on concurrent CrewAI kickoff() calls, regardless of how many
# domains/entities are running in parallel. Per-domain semaphores
# (settings.synthesis.max_parallel_subdomains) only bound concurrency *within*
# one domain; this bounds it across the whole run.
crew_semaphore = threading.Semaphore(settings.llm.max_concurrent_calls)

_RETRYABLE_MARKERS = (
    "429", "rate limit", "rate_limit", "too many requests",
    "timeout", "timed out", "connection", "503", "502", "overloaded",
)


def is_retryable(exc: Exception) -> bool:
    """True if *exc* looks like a transient rate-limit/connection error."""
    text = str(exc).lower()
    return any(marker in text for marker in _RETRYABLE_MARKERS)


def call_with_backoff(fn, *args, max_attempts: int = 3, base_delay: float = 2.0, **kwargs):
    """Call fn(*args, **kwargs), retrying transient errors with exponential backoff.

    Non-retryable exceptions (a real prompt/JSON bug) propagate immediately on
    the first attempt instead of being retried 3 times for nothing.
    """
    last_exc: Exception
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not is_retryable(exc) or attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logger.warning(
                "call_with_backoff: attempt %d/%d failed (%s), retrying in %.1fs",
                attempt + 1, max_attempts, exc, delay,
            )
            time.sleep(delay)
    raise last_exc
