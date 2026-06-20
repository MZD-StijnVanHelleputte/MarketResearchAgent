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
from abc import ABC

from crewai import Agent, Crew, LLM, Task
from pydantic import ValidationError

import models.crewai_patches  # noqa: F401 — must run before any Agent/Crew/Task is built
from config import settings
from core.event_logger import log_event, set_run_context
from core.friendly_names import friendly_domain, friendly_tool
from core.schemas import ChapterDraft
from core.tool_router import async_route
from models.llm_client import LLMClient
from models.llm_retry import call_with_backoff, crew_semaphore
from models.usage import crew_usage, llm_usage, merge_usage
from prompts.domain_agent_prompt import DOMAIN_TASK_TEMPLATE, TOOL_REPAIR_TEMPLATE

logger = logging.getLogger(__name__)


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
            await log_event(
                "progress",
                f"Fetching {friendly_tool(tool_name)} for {friendly_domain(self.DOMAIN)}…",
            )
            async with sem:
                result, usages, error = await self._call_tool_with_repair(tool_name, arguments)
            return tool_name, result, usages, error

        outcomes = await asyncio.gather(*(_run_one(tc) for tc in domain_calls))

        raw_results: list[dict] = []
        tool_errors: list[str] = []
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

        draft.tool_errors = tool_errors
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
                "citations, contradiction_flags"
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
            return ChapterDraft(
                domain=self.DOMAIN,
                plan_id=plan_id,
                text=raw_output,
            )

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
                value_col = f"value ({unit})" if unit else "value"
                rows = [[str(r.get("date", "")), str(r.get("value", ""))] for r in rows_in[:CAP]]
                title = f"{symbol or 'series'} — {len(rows_in)} point(s)"
                datasets.append({
                    "tool": tool, "title": title, "kind": "table",
                    "columns": ["date", value_col], "rows": rows, "row_count": len(rows_in),
                    # Stable identity independent of row count, so the same series
                    # collected by different plans/domains can be deduped reliably.
                    "series_id": f"{tool}:{symbol}",
                })

            elif isinstance(result, dict) and "ticker" in result and "rows" in result:
                ticker = str(result.get("ticker", "") or "")
                period = result.get("period") or ""
                rows_in = [r for r in (result.get("rows") or []) if isinstance(r, dict)]
                columns: list[str] = []
                for r in rows_in:
                    for k in r.keys():
                        if k not in columns:
                            columns.append(k)
                rows = [[str(r.get(c, "")) for c in columns] for r in rows_in[:CAP]]
                title = f"{ticker or 'equity'} {period} — {len(rows_in)} row(s)".replace("  ", " ").strip()
                datasets.append({
                    "tool": tool, "title": title, "kind": "table",
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
                    "kind": "list", "items": items,
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
                    "kind": "list", "items": items,
                })

            elif isinstance(result, dict) and "filings" in result:
                filings = result.get("filings") or []
                items = [{
                    "title": f"{f.get('entity_name', '')} — {f.get('form_type', '')}".strip(" —"),
                    "url": f.get("url") or f.get("filing_url"),
                    "snippet": f.get("file_date", ""),
                } for f in filings[:CAP] if isinstance(f, dict)]
                datasets.append({
                    "tool": tool, "title": f"{len(filings)} filing(s)",
                    "kind": "list", "items": items,
                })

            else:
                summary = json.dumps(result, default=str)[:2000] if result else ""
                datasets.append({
                    "tool": tool, "title": tool,
                    "kind": "summary", "summary": summary,
                })

        return datasets

    def _fallback(self, plan_id: str, raw_results: list[dict]) -> ChapterDraft:
        """Format raw tool results as plain text without an LLM call."""
        lines: list[str] = []
        citations: list[str] = []
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
                    url = a.get("url") or a.get("link")
                    if title or desc:
                        lines.append(f"[{tool}] {title} — {desc}")
                    if url:
                        citations.append(url)
            elif "results" in result:
                for r in result["results"][:3]:
                    title = r.get("title", "")
                    content = r.get("content", r.get("snippet", ""))
                    url = r.get("url") or r.get("link")
                    if title or content:
                        lines.append(f"[{tool}] {title} — {content}")
                    if url:
                        citations.append(url)
            elif "filings" in result:
                for f in result["filings"][:3]:
                    name = f.get("entity_name", "")
                    form = f.get("form_type", "")
                    fdate = f.get("file_date", "")
                    url = f.get("url") or f.get("filing_url")
                    if name or form:
                        lines.append(f"[{tool}] {name} — {form} ({fdate})")
                    if url:
                        citations.append(url)
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
