"""Abstract base class for domain-specific data-collection sub-agents.

Each domain agent:
  1. Filters the plan's tool_calls to those tagged for its domain.
  2. Executes each tool call via async_route.
  3. Passes raw results to a CrewAI agent for structured extraction.
  4. Returns a ChapterDraft (fallback to plain-text if CrewAI fails).

Dependency rules:
  - MAY import from core/schemas.py, core/tool_router.py, config/, models/, prompts/
  - MUST NOT import from core/graph.py
  - MUST NOT call retrieval/ or memory/ directly (graph.py handles ChromaDB writes)
"""
import asyncio
import concurrent.futures
import json
import logging
import re
from abc import ABC

from crewai import Agent, Crew, LLM, Task
from pydantic import ValidationError

import models.crewai_patches  # noqa: F401 — must run before any Agent/Crew/Task is built
from config import settings
from core.event_logger import log_event, set_run_context
from core.friendly_names import friendly_domain, friendly_tool
from core.schemas import ChapterDraft
from core.tool_router import async_route, is_permanent_tool_error
from models.json_repair import extract_json_field
from models.llm_client import LLMClient
from models.llm_retry import call_with_backoff, crew_semaphore
from models.usage import crew_usage, llm_usage, merge_usage
from prompts.domain_agent_prompt import DOMAIN_TASK_TEMPLATE, TOOL_REPAIR_TEMPLATE
from tools.registry import tool_display_name

logger = logging.getLogger(__name__)

# FRED tools whose only required argument is a series_id the planner guessed —
# these get an extra discovery-based recovery step on a permanent "series does
# not exist" error (see _resolve_fred_series_id) instead of failing fast.
_FRED_ID_TOOLS = {"get_macro_indicator", "get_fred_observations"}

# Argument keys, in priority order, that best identify *what* a tool call is
# fetching (e.g. "CAT" for a ticker lookup) — used to make live progress
# labels concrete instead of just naming the tool.
_IDENTIFIER_KEYS = ("ticker", "symbol", "series_id", "query", "commodity", "name", "entity_name")


def _primary_argument(arguments: dict) -> str | None:
    """Best-effort identifier for a tool call's arguments, for progress labels."""
    for key in _IDENTIFIER_KEYS:
        if arguments.get(key):
            return str(arguments[key])
    if len(arguments) == 1:
        return str(next(iter(arguments.values())))
    return None


def _load_json(raw: str) -> dict | None:
    """Parse a JSON object from an LLM response, tolerating markdown code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines() if not line.startswith("```")
        ).strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


class BaseDomainAgent(ABC):
    """Base class for all 7 domain collection sub-agents.

    Subclasses define DOMAIN, ROLE, GOAL, BACKSTORY as class-level strings.
    The run() method is the only public interface.
    """

    DOMAIN: str = ""
    ROLE: str = ""
    GOAL: str = ""
    BACKSTORY: str = ""

    def __init__(self) -> None:
        api_key = getattr(settings, "mistral_api_key", None)
        self._llm = LLM(
            model=f"mistral/{settings.llm.model}",
            api_key=api_key,
            temperature=settings.llm.work_temperature,
        )

    async def run(self, plan: dict, run_id: str) -> ChapterDraft:
        """Execute domain data collection and return a structured ChapterDraft."""
        set_run_context(run_id, stage="collect", domain=self.DOMAIN)
        plan_id = plan.get("plan_id", "unknown")
        # ConsolidatedPlan serialises as "planned_tool_calls"; CandidatePlan uses "tool_calls".
        raw_calls = plan.get("planned_tool_calls") or plan.get("tool_calls") or []
        domain_calls = [tc for tc in raw_calls if tc.get("domain") == self.DOMAIN]

        await log_event(
            "domain_filter",
            f"{self.DOMAIN}: {len(domain_calls)}/{len(raw_calls)} calls matched",
            detail={"plan_id": plan_id, "total_raw": len(raw_calls), "matched": len(domain_calls)},
            level="info" if domain_calls else "warning",
        )

        if not domain_calls:
            return ChapterDraft(
                domain=self.DOMAIN,
                plan_id=plan_id,
                text=f"No tool calls assigned to {self.DOMAIN} in plan {plan_id}.",
            )

        await log_event(
            "progress",
            f"Collecting {friendly_domain(self.DOMAIN)} data from {len(domain_calls)} source(s)…",
        )

        # Execute tool calls concurrently (bounded by max_parallel_tool_calls), collect
        # raw results. Each attempted call is one external API request counted toward
        # the run's api_call_count.
        sem = asyncio.Semaphore(settings.react.max_parallel_tool_calls)

        async def _run_one(tc: dict) -> tuple[str, dict | None, list[dict], str | None]:
            tool_name = tc.get("tool", "")
            # PlannedToolCall uses "params"; CandidatePlan tool_calls use "arguments".
            arguments = tc.get("params") or tc.get("arguments") or {}
            identifier = _primary_argument(arguments)
            suffix = f": {identifier}" if identifier else ""
            await log_event(
                "progress",
                f"Fetching {friendly_tool(tool_name)}{suffix} for {friendly_domain(self.DOMAIN)}…",
            )
            async with sem:
                result, usages, error = await self._call_tool_with_repair(tool_name, arguments)
            return tool_name, result, usages, error

        outcomes = await asyncio.gather(*(_run_one(tc) for tc in domain_calls))

        raw_results: list[dict] = []
        tool_errors: list[str] = []
        failed_tools: list[dict] = []
        repair_usages: list[dict] = []
        tool_calls_made = len(domain_calls)
        for tool_name, result, usages, error in outcomes:
            repair_usages.extend(usages)
            if result is not None:
                raw_results.append({"tool": tool_name, "result": result})
            else:
                logger.warning(
                    "domain_agent %s: tool %s failed: %s", self.DOMAIN, tool_name, error
                )
                tool_errors.append(f"{self.DOMAIN}/{tool_name}: {error}")
                failed_tools.append({
                    "tool": tool_name,
                    "tool_display": tool_display_name(tool_name),
                    "reason": str(error),
                })

        tool_usage = {"prompt_tokens": 0, "completion_tokens": 0, "requests": tool_calls_made}

        # Synthesise raw results via CrewAI; fall back to plain-text formatting
        try:
            draft = self._run_crew(plan_id, raw_results)
        except Exception as exc:
            logger.warning(
                "domain_agent %s: CrewAI failed (%s), using fallback", self.DOMAIN, exc
            )
            draft = self._fallback(plan_id, raw_results)
            tool_errors.append(f"{self.DOMAIN}/synthesis: {exc}")
            failed_tools.append({
                "tool": "synthesis",
                "tool_display": "Synthesis",
                "reason": str(exc),
            })

        draft.tool_errors = tool_errors
        draft.failed_tools = failed_tools
        draft.datasets = self._to_datasets(raw_results)
        draft.usage = merge_usage(draft.usage, tool_usage, *repair_usages)
        return draft

    async def _call_tool_with_repair(
        self, tool_name: str, arguments: dict
    ) -> tuple[dict | None, list[dict], str | None]:
        """Attempt a tool call; on failure, use an LLM to interpret the error and adapt
        the arguments, retrying up to settings.react.tool_repair_max_attempts times.

        Returns (result_or_None, repair_usage_dicts, last_error_string).
        """
        repair_usages: list[dict] = []
        try:
            return await async_route(tool_name, arguments), repair_usages, None
        except Exception as exc:
            last_error = str(exc)
            # A permanent client error (e.g. FRED 400 "the series does not exist") can
            # never be fixed by retrying with adapted arguments — skip the LLM repair
            # loop and fail fast instead of burning several LLM round-trips.
            if is_permanent_tool_error(exc):
                series_id = arguments.get("series_id")
                if tool_name in _FRED_ID_TOOLS and series_id:
                    resolved_id = await self._resolve_fred_series_id(series_id)
                    if resolved_id and resolved_id != series_id:
                        new_args = {**arguments, "series_id": resolved_id}
                        try:
                            result = await async_route(tool_name, new_args)
                            await log_event(
                                "progress",
                                f"Resolved FRED series '{series_id}' → "
                                f"'{resolved_id}' via discovery.",
                                detail={"original": series_id, "resolved": resolved_id},
                            )
                            return result, repair_usages, None
                        except Exception as exc2:
                            last_error = str(exc2)
                return None, repair_usages, last_error

        current_args = dict(arguments)
        for _ in range(settings.react.tool_repair_max_attempts):
            friendly = friendly_tool(tool_name)
            await log_event(
                "progress",
                f"That didn't work for {friendly} — retrying…",
                detail={"error": last_error, "args": current_args},
            )
            decision, usage = await self._repair_decision(tool_name, current_args, last_error)
            if usage:
                repair_usages.append(usage)
            if not decision or decision.get("action") != "retry":
                break
            new_args = decision.get("arguments")
            if not isinstance(new_args, dict) or not new_args or new_args == current_args:
                break
            current_args = new_args
            try:
                result = await async_route(tool_name, current_args)
                await log_event(
                    "progress",
                    f"Retry succeeded for {friendly}.",
                    detail={"args": current_args, "reason": decision.get("reason", "")},
                )
                return result, repair_usages, None
            except Exception as exc:
                last_error = str(exc)

        return None, repair_usages, last_error

    async def _repair_decision(
        self, tool_name: str, arguments: dict, error: str
    ) -> tuple[dict | None, dict]:
        """Ask the LLM whether/how to adapt a failed tool call. Returns (decision, usage)."""
        prompt = TOOL_REPAIR_TEMPLATE.format(
            tool_name=tool_name,
            arguments_json=json.dumps(arguments, default=str),
            error=error,
        )
        try:
            response = await LLMClient().acomplete(
                [{"role": "user", "content": prompt}], temperature=0.0
            )
        except Exception as exc:
            logger.warning("domain_agent %s: tool repair LLM failed: %s", self.DOMAIN, exc)
            return None, {}
        return _load_json(response.content or ""), llm_usage(response.usage)

    @staticmethod
    async def _resolve_fred_series_id(series_id: str) -> str | None:
        """Recover a real FRED series_id when the planner guessed one that doesn't
        exist, by querying FRED's own discovery endpoints: series/search first
        (free-text, https://fred.stlouisfed.org/docs/api/fred/series_search.html),
        falling back to tags/series (https://fred.stlouisfed.org/docs/api/fred/tags_series.html)
        if the search comes up empty. Returns None if neither finds a match."""
        tokens = [t for t in re.split(r"[_\-]+", series_id) if t]
        query = " ".join(tokens) or series_id

        try:
            search_result = await async_route(
                "search_fred_series", {"search_text": query, "limit": 5}
            )
        except Exception:
            search_result = None
        candidates = (search_result or {}).get("results") or []
        if candidates:
            return candidates[0].get("series_id")

        tag_names = ";".join(t.lower() for t in tokens if len(t) > 2)
        if not tag_names:
            return None
        try:
            tags_result = await async_route(
                "get_fred_series_by_tags", {"tag_names": tag_names, "limit": 5}
            )
        except Exception:
            return None
        candidates = (tags_result or {}).get("results") or []
        return candidates[0].get("series_id") if candidates else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_crew(self, plan_id: str, raw_results: list[dict]) -> ChapterDraft:
        context_json = json.dumps(raw_results, indent=2, default=str)
        prompt = DOMAIN_TASK_TEMPLATE.format(
            domain=self.DOMAIN,
            plan_id=plan_id,
            context_json=context_json,
        )

        agent = Agent(
            role=self.ROLE,
            goal=self.GOAL,
            backstory=self.BACKSTORY,
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(
            description=prompt,
            expected_output=(
                "A JSON object with keys: domain, plan_id, text, figures, "
                "contradiction_flags"
            ),
            agent=agent,
        )
        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        def _run_crew() -> object:
            with crew_semaphore:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(crew.kickoff).result()

        result = call_with_backoff(_run_crew)
        draft = self._parse_result(str(result), plan_id)
        # Citation identity is always code-derived, never LLM-authored — an LLM
        # asked to "list source identifiers" will happily cite a tool/agent name
        # or scaffolding text instead of a real source, so its output (if any)
        # is discarded in favor of deterministic extraction from raw_results.
        draft.citations = self._extract_citations(raw_results)
        draft.usage = merge_usage(draft.usage, crew_usage(result))
        return draft

    def _parse_result(self, raw_output: str, plan_id: str) -> ChapterDraft:
        # Strip markdown fences if present
        text = raw_output.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines
                if not line.startswith("```")
            ).strip()
        try:
            data = json.loads(text)
            data.setdefault("domain", self.DOMAIN)
            data.setdefault("plan_id", plan_id)
            return ChapterDraft.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Full envelope didn't parse — recover at least the prose so a malformed
        # sibling key (e.g. a stray quote in "citations") doesn't leak raw JSON
        # into the report. If even that fails, let the caller's fallback (which
        # preserves figures/citations from the raw tool results) take over.
        recovered_text = extract_json_field(raw_output, "text")
        if recovered_text is None:
            raise ValueError(
                f"could not parse domain agent response for {self.DOMAIN}: {raw_output[:200]!r}"
            )
        return ChapterDraft(
            domain=self.DOMAIN,
            plan_id=plan_id,
            text=recovered_text,
        )

    @staticmethod
    def _citation_key(citation: dict) -> tuple:
        """Dedup key for a citation dict: by URL when present, else title+publisher."""
        url = (citation.get("url") or "").strip().lower()
        if url:
            return ("url", url)
        return ("title", (citation.get("title") or "").strip().lower(),
                (citation.get("publisher") or "").strip().lower())

    @staticmethod
    def _extract_citations(raw_results: list[dict]) -> list[dict]:
        """Deterministically derive structured citations from raw tool results.

        Mirrors the shape detection in _to_datasets()/_fallback() but emits
        {"id": None, "title", "url", "publisher"} records instead of free text,
        so the LLM is never the source of a citation's identity (see _run_crew).
        Tool results with no document/URL (commodity/FRED/financials series) get
        a real human title via tool_display_name() — never the raw tool name.
        """
        citations: list[dict] = []
        seen: set = set()

        def _add(title: str, url: str | None = None, publisher: str | None = None) -> None:
            if not title:
                return
            citation = {"id": None, "title": title, "url": url or None, "publisher": publisher or None}
            key = BaseDomainAgent._citation_key(citation)
            if key not in seen:
                seen.add(key)
                citations.append(citation)

        for entry in raw_results:
            tool = entry.get("tool", "") or "unknown"
            result = entry.get("result", {})
            if not isinstance(result, dict):
                continue

            if "articles" in result:
                for a in result["articles"] or []:
                    if isinstance(a, dict):
                        _add(a.get("title", ""), a.get("url") or a.get("link"), a.get("source"))
            elif "results" in result:
                for r in result["results"] or []:
                    if isinstance(r, dict):
                        _add(r.get("title", ""), r.get("url") or r.get("link"))
            elif "filings" in result:
                for f in result["filings"] or []:
                    if isinstance(f, dict):
                        name = f.get("entity_name", "")
                        form = f.get("form_type", "")
                        title = f"{name} — {form}".strip(" —")
                        _add(title, f.get("url") or f.get("filing_url"), "SEC EDGAR")
            elif "technical_report" in result:
                tr = result.get("technical_report") or {}
                company = tr.get("company_name", "")
                form = tr.get("form_type", "")
                title = tr.get("exhibit_name") or f"{company} — {form} Exhibit 96".strip(" —")
                _add(title, tr.get("exhibit_url"), "SEC EDGAR")
            elif "report" in result and "citations" in result:
                # web_research (Tavily) ResearchReport shape: AI-synthesized report
                # backed by a flat list of source URLs, with no per-URL title.
                for url in result.get("citations") or []:
                    if isinstance(url, str) and url:
                        _add(url, url)
            else:
                # Data-only tool results (commodity series, FRED, financials, masterdata)
                # carry no document/URL — title is a real provider-aware label, never
                # the raw tool function name.
                _add(tool_display_name(tool), None, result.get("source"))

        return citations

    @staticmethod
    def _to_datasets(raw_results: list[dict]) -> list[dict]:
        """Normalize raw tool results into Gate-2 datasets (table / list / summary).

        Mirrors the shape detection in _fallback() so the human can review the actual
        numbers and sources collected per domain. Rows/items are capped to bound state
        size; the true count is preserved in row_count.
        """
        CAP = 500
        datasets: list[dict] = []
        for entry in raw_results:
            tool = entry.get("tool", "") or "unknown"
            result = entry.get("result", {})

            if isinstance(result, dict) and "symbol" in result and (
                "latest" in result or "rows" in result
            ):
                symbol = str(result.get("symbol", "") or "")
                unit = result.get("unit") or ""
                rows_in = [
                    r for r in (result.get("rows") or [])
                    if isinstance(r, dict) and r.get("value") is not None
                ]
                if not rows_in:
                    latest = result.get("latest") or {}
                    if isinstance(latest, dict) and latest.get("value") is not None:
                        rows_in = [latest]
                # Sort oldest -> newest and keep the most recent CAP points, so a
                # truncated series still reaches up to the latest available date
                # instead of being cut off at the oldest CAP rows.
                rows_in.sort(key=lambda r: str(r.get("date", "")))
                value_col = f"value ({unit})" if unit else "value"
                rows = [[str(r.get("date", "")), str(r.get("value", ""))] for r in rows_in[-CAP:]]
                title = f"{symbol or 'series'} — {len(rows_in)} point(s)"
                datasets.append({
                    "tool": tool, "title": title, "kind": "table",
                    "data_type": "numeric_series", "label": symbol or "series",
                    "count": len(rows_in),
                    "columns": ["date", value_col], "rows": rows, "row_count": len(rows_in),
                    # Stable identity independent of row count, so the same series
                    # collected by different plans/domains can be deduped reliably.
                    "series_id": f"{tool}:{symbol}",
                })

            elif isinstance(result, dict) and "ticker" in result and "rows" in result:
                ticker = str(result.get("ticker", "") or "")
                period = result.get("period") or ""
                rows_in = [r for r in (result.get("rows") or []) if isinstance(r, dict)]
                date_key = next((k for k in (rows_in[0].keys() if rows_in else []) if "date" in k.lower()), None)
                if date_key:
                    # Sort oldest -> newest and keep the most recent CAP rows, so a
                    # truncated series still reaches up to the latest available date.
                    rows_in.sort(key=lambda r: str(r.get(date_key, "")))
                    rows_capped = rows_in[-CAP:]
                else:
                    rows_capped = rows_in[:CAP]
                columns: list[str] = []
                for r in rows_capped:
                    for k in r.keys():
                        if k not in columns:
                            columns.append(k)
                rows = [[str(r.get(c, "")) for c in columns] for r in rows_capped]
                title = f"{ticker or 'equity'} {period} — {len(rows_in)} row(s)".replace("  ", " ").strip()
                datasets.append({
                    "tool": tool, "title": title, "kind": "table",
                    "data_type": "financials",
                    "label": f"{ticker} {period}".strip() or "equity",
                    "count": len(rows_in),
                    "columns": columns, "rows": rows, "row_count": len(rows_in),
                    "series_id": f"{tool}:{ticker}:{period}",
                })

            elif isinstance(result, dict) and "articles" in result:
                arts = result.get("articles") or []
                items = [{
                    "title": a.get("title", ""),
                    "url": a.get("url") or a.get("link"),
                    "snippet": a.get("description", ""),
                } for a in arts[:CAP] if isinstance(a, dict)]
                datasets.append({
                    "tool": tool, "title": f"{len(arts)} article(s)",
                    "kind": "list", "data_type": "articles", "label": "",
                    "count": len(arts), "items": items,
                })

            elif isinstance(result, dict) and "results" in result:
                res = result.get("results") or []
                items = [{
                    "title": r.get("title", ""),
                    "url": r.get("url") or r.get("link"),
                    "snippet": r.get("content", r.get("snippet", "")),
                } for r in res[:CAP] if isinstance(r, dict)]
                datasets.append({
                    "tool": tool, "title": f"{len(res)} result(s)",
                    "kind": "list", "data_type": "web_results", "label": "",
                    "count": len(res), "items": items,
                })

            elif isinstance(result, dict) and "filings" in result:
                filings = result.get("filings") or []
                items = [{
                    "title": f"{f.get('entity_name', '')} — {f.get('form_type', '')}".strip(" —"),
                    "url": f.get("url") or f.get("filing_url"),
                    "snippet": f.get("file_date", ""),
                } for f in filings[:CAP] if isinstance(f, dict)]
                first_entity = next(
                    (str(f.get("entity_name", "")).strip() for f in filings
                     if isinstance(f, dict) and f.get("entity_name")), "")
                datasets.append({
                    "tool": tool, "title": f"{len(filings)} filing(s)",
                    "kind": "list", "data_type": "filings", "label": first_entity,
                    "count": len(filings), "items": items,
                })

            elif isinstance(result, dict) and "technical_report" in result:
                tr = result.get("technical_report") or {}
                title = f"{tr.get('company_name', '')} — {tr.get('form_type', '')} Exhibit 96".strip(" —")
                datasets.append({
                    "tool": tool, "title": title, "kind": "summary",
                    "data_type": "document",
                    "label": str(tr.get("company_name", "")).strip(), "count": 1,
                    "summary": tr.get("excerpt", ""),
                    "items": [{
                        "title": tr.get("exhibit_name", ""),
                        "url": tr.get("exhibit_url"),
                        "snippet": tr.get("filing_date", ""),
                    }],
                })

            else:
                summary = json.dumps(result, default=str)[:2000] if result else ""
                datasets.append({
                    "tool": tool, "title": tool,
                    "kind": "summary", "data_type": "data", "label": "", "count": 1,
                    "summary": summary,
                })

        return datasets

    def _fallback(self, plan_id: str, raw_results: list[dict]) -> ChapterDraft:
        """Format raw tool results as plain text without an LLM call."""
        lines: list[str] = []
        citations = self._extract_citations(raw_results)
        figures: dict[str, str] = {}
        for entry in raw_results:
            tool = entry.get("tool", "")
            result = entry.get("result", {})
            if isinstance(result, dict) and "symbol" in result and (
                "latest" in result or "rows" in result
            ):
                # Commodity/structured price result (CommodityResult shape).
                symbol = str(result.get("symbol", "")).lower()
                unit = result.get("unit") or ""
                latest = result.get("latest") or {}
                rows = result.get("rows") or []
                if isinstance(latest, dict) and latest.get("value") is not None:
                    metric = f"{symbol}_latest" + (f"_{unit}" if unit else "")
                    figures[metric] = str(latest["value"])
                    lines.append(
                        f"[{tool}] {symbol or 'commodity'} latest: {latest['value']} "
                        f"{unit} (as of {latest.get('date', 'n/a')})".strip()
                    )
                recent = [
                    r for r in rows[:4]
                    if isinstance(r, dict) and r.get("value") is not None
                ]
                if recent:
                    series = ", ".join(f"{r.get('date', '')}: {r['value']}" for r in recent)
                    lines.append(f"[{tool}] {symbol or 'commodity'} recent — {series}")
            elif "articles" in result:
                for a in result["articles"][:3]:
                    title = a.get("title", "")
                    desc = a.get("description", "")
                    if title or desc:
                        lines.append(f"[{tool}] {title} — {desc}")
            elif "results" in result:
                for r in result["results"][:3]:
                    title = r.get("title", "")
                    content = r.get("content", r.get("snippet", ""))
                    if title or content:
                        lines.append(f"[{tool}] {title} — {content}")
            elif "filings" in result:
                for f in result["filings"][:3]:
                    name = f.get("entity_name", "")
                    form = f.get("form_type", "")
                    fdate = f.get("file_date", "")
                    if name or form:
                        lines.append(f"[{tool}] {name} — {form} ({fdate})")
            elif "technical_report" in result:
                tr = result.get("technical_report") or {}
                lines.append(
                    f"[{tool}] {tr.get('company_name', '')} {tr.get('form_type', '')} "
                    f"Exhibit 96 ({tr.get('filing_date', '')}): {tr.get('excerpt', '')[:300]}"
                )
            else:
                summary = str(result)[:300]
                if summary:
                    lines.append(f"[{tool}] {summary}")

        text = "\n".join(lines) if lines else f"No data collected for {self.DOMAIN}."
        return ChapterDraft(
            domain=self.DOMAIN,
            plan_id=plan_id,
            text=text,
            figures=figures,
            citations=citations,
        )
