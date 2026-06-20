"""Unit test for the crewai cache_breakpoint workaround (models/crewai_patches.py).

See models/crewai_patches.py for why: crewai 1.14.6 stamps every system/user
message with cache_breakpoint=True, but never strips it for Mistral, so the
API rejects the request with extra_forbidden. This patch makes the marker a
no-op project-wide (safe since this project only ever talks to Mistral).
"""
import crewai.llms.cache as crewai_cache

import models.crewai_patches  # noqa: F401 — applies the patch on import


def test_mark_cache_breakpoint_is_a_noop_after_patch():
    message = {"role": "system", "content": "hello"}
    result = crewai_cache.mark_cache_breakpoint(message)
    assert "cache_breakpoint" not in result
    assert result == message


def test_patch_returns_the_same_object_unchanged():
    message = {"role": "user", "content": "what's the copper outlook?"}
    result = crewai_cache.mark_cache_breakpoint(message)
    assert result is message
