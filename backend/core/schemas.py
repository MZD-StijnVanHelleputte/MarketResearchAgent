"""Shared Pydantic schemas for inter-module data transfer.

These types are used by agents/, core/merger.py, and core/graph.py.
This module must NOT import from core/graph.py to avoid circular imports.
"""
from pydantic import BaseModel, Field


class ChapterDraft(BaseModel):
    """Raw chapter produced by a single domain sub-agent for one survivor plan."""

    domain: str
    plan_id: str
    text: str
    figures: dict[str, str] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    contradiction_flags: list[str] = Field(default_factory=list)
    # Tool calls that failed during collection (e.g. "news_search: HTTP 429 ..."),
    # surfaced so failures are visible instead of only logged to stdout.
    tool_errors: list[str] = Field(default_factory=list)
    # Structured form of the failures above for the UI: {tool, tool_display, reason}.
    # Lets the frontend render "tried X — failed: reason" in red without parsing strings.
    failed_tools: list[dict] = Field(default_factory=list)
    # Normalized raw tool results for the Gate 2 data review (tables/lists/summaries).
    datasets: list[dict] = Field(default_factory=list)
    # Token usage for the live cost counter: {prompt_tokens, completion_tokens, requests}
    usage: dict = Field(default_factory=dict)


class MergedChapter(BaseModel):
    """Chapter produced by merging ChapterDrafts from all survivor plans for a domain."""

    domain: str
    text: str
    figures: dict[str, str] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    contradiction_flags: list[str] = Field(default_factory=list)
    source_plan_ids: list[str] = Field(default_factory=list)
    # Carried over from the authoritative ChapterDraft so structured tables
    # (e.g. equity history, financials) reach the PDF instead of being dropped.
    datasets: list[dict] = Field(default_factory=list)


class SubChapter(BaseModel):
    """Tier-1 leaf analysis for one entity/subdomain within a domain.

    Produced during synthesis by decomposing a MergedChapter into its
    constituent entities (e.g. each competitor, each commodity). Several
    SubChapters roll up into the domain's chapter text (Tier 2).
    """

    domain: str
    subdomain_key: str        # stable key, e.g. "CAT", "copper", "Sandvik"
    subdomain_label: str      # human label, e.g. "Caterpillar Inc."
    text: str
    figures: dict[str, str] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    contradiction_flags: list[str] = Field(default_factory=list)
    datasets: list[dict] = Field(default_factory=list)
    # Token usage for the live cost counter: {prompt_tokens, completion_tokens, requests}
    usage: dict = Field(default_factory=dict)
