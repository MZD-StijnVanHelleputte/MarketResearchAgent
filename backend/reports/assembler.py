"""Assembles final graph state into a ReportDraft for PDF rendering."""
from pydantic import BaseModel

from core.domains import ownership_order

# Chapter order in the report follows domain ownership priority, with a trailing
# "supplementary" bucket for anything not tied to a domain.
_DOMAIN_ORDER = [*ownership_order(), "supplementary"]

_DOMAIN_ORDER_INDEX = {d: i for i, d in enumerate(_DOMAIN_ORDER)}


class ReportDraft(BaseModel):
    run_id: str
    query: str
    exec_summary: str
    chapters: list[dict]    # [{domain, text}] ordered by domain priority
    merge_log: list[str]
    warnings: list[str]
    # {id: {"title", "url", "publisher"}} — every unique source across all domains,
    # keyed by the global id assigned in core/merger.py::assign_global_citation_ids.
    citation_registry: dict = {}


class Assembler:
    @staticmethod
    def assemble(final_state: dict, run_id: str, query: str) -> ReportDraft:
        chapters = list(final_state.get("synthesis_chapters") or [])

        # Order by canonical domain priority; unknown domains go last
        chapters.sort(
            key=lambda ch: _DOMAIN_ORDER_INDEX.get(
                ch.get("domain", ""), len(_DOMAIN_ORDER)
            )
        )

        return ReportDraft(
            run_id=run_id,
            query=query,
            exec_summary=final_state.get("exec_summary") or "",
            chapters=chapters,
            merge_log=list(final_state.get("merge_log") or []),
            warnings=list(final_state.get("warnings") or []),
            citation_registry=dict(final_state.get("citation_registry") or {}),
        )
