"""CrewAI-based synthesis sub-agent.

Takes a MergedChapter (output of core/merger.py) plus pre-fetched context
(retrieved Chroma chunks and SQLite figures) and produces polished chapter prose.

Dependency rules:
  - MAY import from core/schemas.py, config/, prompts/
  - MUST NOT import from core/graph.py or memory/ directly
  - Receives retrieved_chunks and sqlite_figures as arguments from synthesize_node
"""
import concurrent.futures
import json
import logging

from crewai import Agent, Crew, LLM, Task

import models.crewai_patches  # noqa: F401 — must run before any Agent/Crew/Task is built
from config import settings
from core.schemas import MergedChapter
from models.llm_retry import call_with_backoff, crew_semaphore
from models.usage import crew_usage
from prompts.domain_agent_prompt import SYNTHESIS_TASK_TEMPLATE
from prompts.synthesize_prompt import DOMAIN_ROLLUP_TEMPLATE, SUBDOMAIN_SYNTHESIS_TEMPLATE

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """Wraps the per-domain synthesis LLM call in a CrewAI agent."""

    def __init__(self) -> None:
        api_key = getattr(settings, "mistral_api_key", None)
        self._llm = LLM(
            model=f"mistral/{settings.llm.model}",
            api_key=api_key,
            temperature=settings.llm.work_temperature,
        )

    def run(
        self,
        domain: str,
        merged_chapter: MergedChapter,
        retrieved_chunks: list,
        sqlite_figures: list[dict],
        query: str = "",
    ) -> dict:
        """Produce a polished chapter dict compatible with synthesis_chapters format.

        Returns {"domain": str, "text": str}.
        Falls back to formatting the merged chapter text directly on CrewAI failure.
        """
        try:
            return self._run_crew(domain, merged_chapter, retrieved_chunks, sqlite_figures, query)
        except Exception as exc:
            logger.warning("synthesis_agent %s: CrewAI failed (%s), using fallback", domain, exc)
            return self._fallback(domain, merged_chapter)

    def run_subchapter(
        self,
        domain: str,
        subdomain_key: str,
        subdomain_label: str,
        figures: dict[str, str],
        datasets: list[dict],
        citations: list[str],
        retrieved_chunks: list,
        query: str = "",
    ) -> dict:
        """Tier-1: write a focused analysis of one entity within *domain*.

        Returns a SubChapter-shaped dict. Falls back to a minimal evidence
        summary on CrewAI failure so the rollup always has leaves to work with.
        """
        evidence_json = json.dumps(
            {"figures": figures, "datasets": datasets, "citations": citations},
            indent=2, default=str,
        )
        prompt = SUBDOMAIN_SYNTHESIS_TEMPLATE.format(
            domain=domain,
            subdomain_label=subdomain_label,
            research_question=query,
            evidence_json=evidence_json,
            retrieved_chunks_text=self._format_chunks(retrieved_chunks),
            min_words=settings.synthesis.subdomain_min_words,
            max_words=settings.synthesis.subdomain_max_words,
        )
        synthesis_error: str | None = None
        try:
            text, usage = self._kickoff(
                prompt,
                role="Entity Intelligence Analyst",
                goal=f"Write a focused {subdomain_label} analysis for the {domain} domain.",
                expected_output='A JSON object with keys "domain", "subdomain_label", "text".',
            )
            text = self._append_figures(text, figures)
        except Exception as exc:
            synthesis_error = str(exc)
            logger.warning(
                "synthesis_agent %s/%s: subchapter CrewAI failed (%s), using fallback",
                domain, subdomain_key, exc,
            )
            text = self._append_figures(
                f"Limited evidence collected for {subdomain_label}.", figures
            )
            usage = {}
        return {
            "domain": domain,
            "subdomain_key": subdomain_key,
            "subdomain_label": subdomain_label,
            "text": text,
            "figures": dict(figures),
            "citations": list(citations),
            "datasets": list(datasets),
            "usage": usage,
            # Real failure reason, if synthesis fell back to a placeholder — surfaced
            # by the completeness gate so it reaches the report's warnings appendix
            # instead of only being logged to stdout.
            "synthesis_error": synthesis_error,
        }

    def run_rollup(
        self,
        domain: str,
        merged_chapter: MergedChapter,
        subchapters: list[dict],
        query: str = "",
    ) -> dict:
        """Tier-2: synthesise per-entity analyses into the domain chapter."""
        subchapters_text = "\n\n".join(
            f"### {sc.get('subdomain_label', sc.get('subdomain_key', '?'))}\n{sc.get('text', '')}"
            for sc in subchapters
        ) or "No entity analyses available."
        prompt = DOMAIN_ROLLUP_TEMPLATE.format(
            domain=domain,
            research_question=query,
            subchapters_text=subchapters_text,
            min_words=settings.synthesis.rollup_min_words,
            max_words=settings.synthesis.rollup_max_words,
        )
        rollup_error: str | None = None
        try:
            text, usage = self._kickoff(
                prompt,
                role="Senior Market Intelligence Writer",
                goal=f"Write the {domain} landscape chapter for Komatsu's intelligence brief.",
                expected_output='A JSON object with keys "domain" and "text".',
            )
            text = self._append_figures(text, merged_chapter.figures)
        except Exception as exc:
            rollup_error = str(exc)
            logger.warning(
                "synthesis_agent %s: rollup CrewAI failed (%s), using fallback", domain, exc
            )
            # Concatenate the leaf analyses so the chapter is never empty.
            text = self._append_figures(subchapters_text, merged_chapter.figures)
            usage = {}
        return {
            "domain": domain, "text": text, "figures": dict(merged_chapter.figures),
            "usage": usage, "synthesis_error": rollup_error,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_crew(
        self,
        domain: str,
        merged_chapter: MergedChapter,
        retrieved_chunks: list,
        sqlite_figures: list[dict],
        query: str = "",
    ) -> dict:
        merged_json = merged_chapter.model_dump_json(indent=2)
        prompt = SYNTHESIS_TASK_TEMPLATE.format(
            domain=domain,
            research_question=query,
            merged_chapter_json=merged_json,
            retrieved_chunks_text=self._format_chunks(retrieved_chunks),
        )
        text, usage = self._kickoff(
            prompt,
            role="Senior Market Intelligence Writer",
            goal=f"Write the final polished {domain} chapter for Komatsu's intelligence brief.",
            expected_output='A JSON object with keys "domain" and "text".',
        )
        text = self._append_figures(text, merged_chapter.figures)
        return {
            "domain": domain,
            "text": text,
            "figures": dict(merged_chapter.figures),
            "usage": usage,
        }

    def _kickoff(
        self,
        prompt: str,
        role: str,
        goal: str,
        expected_output: str,
    ) -> tuple[str, dict]:
        """Run a single-task CrewAI crew and return (parsed_text, usage)."""
        agent = Agent(
            role=role,
            goal=goal,
            backstory=(
                "You are a senior analyst at a top strategy consultancy specialising in "
                "industrial equipment markets. You write concise, evidence-backed briefs "
                "that help Komatsu executives make fast, confident decisions."
            ),
            llm=self._llm,
            verbose=False,
            allow_delegation=False,
        )
        task = Task(description=prompt, expected_output=expected_output, agent=agent)
        crew = Crew(agents=[agent], tasks=[task], verbose=False)

        def _run_crew() -> object:
            with crew_semaphore:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(crew.kickoff).result()

        crew_result = call_with_backoff(_run_crew)
        usage = crew_usage(crew_result)
        result = str(crew_result).strip()

        # Strip markdown fences
        if result.startswith("```"):
            result = "\n".join(
                line for line in result.splitlines() if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(result)
            text = data.get("text", result)
        except (json.JSONDecodeError, KeyError, AttributeError):
            text = result
        return text, usage

    @staticmethod
    def _format_chunks(retrieved_chunks: list) -> str:
        """Format retrieved chunks as readable text for a synthesis prompt."""
        chunk_lines: list[str] = []
        for chunk in retrieved_chunks[:5]:
            text = getattr(chunk, "text", None) or (chunk.get("text", "") if isinstance(chunk, dict) else "")
            source = getattr(chunk, "source", None) or (chunk.get("source", "") if isinstance(chunk, dict) else "")
            if text:
                chunk_lines.append(f"[{source}] {text[:400]}")
        return "\n".join(chunk_lines) if chunk_lines else "No additional context available."

    @staticmethod
    def _append_figures(text: str, figures: dict) -> str:
        """Append a 'Key figures' block only for figures the prose doesn't already mention.

        This is a safety net (numbers must survive even if the LLM drops them), not a
        restatement of everything the model already wove into the narrative — figure
        values (e.g. "$64.8B", "4.12") are distinctive enough that their presence in
        `text` reliably means they were already incorporated.
        """
        if not figures:
            return text
        if "**Key figures**" in text:
            return text
        missing = {k: v for k, v in figures.items() if str(v) not in text}
        if not missing:
            return text
        fig_lines = "\n".join(f"- {k}: {v}" for k, v in missing.items())
        return f"{text}\n\n**Key figures**\n{fig_lines}"

    @classmethod
    def _fallback(cls, domain: str, merged_chapter: MergedChapter) -> dict:
        """Return the merged chapter text as-is without additional synthesis."""
        parts = [merged_chapter.text]
        if merged_chapter.contradiction_flags:
            parts.append("Conflicting signals:\n" + "\n".join(merged_chapter.contradiction_flags))
        text = cls._append_figures("\n\n".join(parts), merged_chapter.figures)
        return {
            "domain": domain,
            "text": text,
            "figures": dict(merged_chapter.figures),
            "usage": {},
        }
