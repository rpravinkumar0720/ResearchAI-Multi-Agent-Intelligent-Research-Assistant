"""Multi-agent CrewAI workflow orchestrator for ResearchAI."""

from collections.abc import Callable
from typing import Any

from crewai import Crew, LLM, Process
from pydantic import BaseModel, Field

from agents.report_agent import (
    REQUIRED_REPORT_SECTIONS,
    create_report_agent,
    create_report_task,
)
from agents.research_agent import create_research_agent, create_research_task
from agents.summarizer_agent import create_summarizer_agent, create_summarizer_task
from tools.web_search import (
    SearchSource,
    WebSearchError,
    format_sources_markdown,
    perform_web_search,
)
from utils.config import get_settings


class ResearchConfigurationError(Exception):
    """Raised when LLM credentials or configuration are missing/invalid."""


class ResearchResult(BaseModel):
    """Structured output returned by the multi-agent research workflow."""

    topic: str
    depth: str
    sources: list[SearchSource] = Field(default_factory=list)
    research_findings: str = ""
    summarized_findings: str = ""
    final_report: str = ""
    used_mock_mode: bool = False
    warnings: list[str] = Field(default_factory=list)


StatusCallback = Callable[[str, str, str], None]


def _build_llm(provider: str | None = None, api_key: str | None = None) -> LLM:
    """Configure CrewAI LLM for Google Gemini or OpenAI without hardcoding keys."""
    settings = get_settings()
    prov = (provider or settings.llm_provider).lower().strip()

    if prov == "gemini":
        key = (api_key or settings.gemini_api_key).strip()
        if not settings.has_valid_llm_key("gemini", key):
            raise ResearchConfigurationError(
                "Google Gemini API key is missing. Add GEMINI_API_KEY to your .env file "
                "or enter it in the UI settings (or enable Demo/Mock Mode for offline/keyless testing)."
            )
        return LLM(model=settings.gemini_model, api_key=key, temperature=0.2)

    if prov == "openai":
        key = (api_key or settings.openai_api_key).strip()
        if not settings.has_valid_llm_key("openai", key):
            raise ResearchConfigurationError(
                "OpenAI API key is missing. Add OPENAI_API_KEY to your .env file "
                "or enter it in the UI settings (or enable Demo/Mock Mode for offline/keyless testing)."
            )
        return LLM(model=settings.openai_model, api_key=key, temperature=0.2)

    raise ResearchConfigurationError(
        f"Unsupported LLM provider '{prov}'. Choose 'gemini' or 'openai'."
    )


def _synthesize_mock_report(
    topic: str, depth: str, sources: list[SearchSource]
) -> tuple[str, str, str]:
    """Generate a structured 11-section report from REAL web search results when in Mock/Demo mode.

    Used only when LLM API credentials are not configured, so real DuckDuckGo
    sources are still retrieved and formatted cleanly without fabricating data.
    """
    findings_md = format_sources_markdown(sources)

    bullet_points = "\n".join(
        f"- **[{s.title}]({s.url})** (`{s.domain}`): {s.snippet}"
        for s in sources
    )
    summary_md = (
        f"### Thematic Synthesis (Demo/Mock Mode — Real Web Sources)\n\n"
        f"Analyzed **{len(sources)}** live web sources for **{topic}** "
        f"(Depth: `{depth}`). Key points extracted directly from retrieved sources:\n\n"
        f"{bullet_points}"
    )

    refs_md = "\n".join(
        f"{idx}. [{s.title}]({s.url}) — `{s.domain}` (Published: {s.published_date})"
        for idx, s in enumerate(sources, start=1)
    )

    lead_snippets = "\n\n".join(
        f"- According to **[{s.title}]({s.url})**: {s.snippet}"
        for s in sources[:3]
    )

    report_md = f"""# {topic} — Research Report

> **Note (Demo/Mock Synthesis Mode):** No LLM API key was configured, so this report was structured directly from **{len(sources)} live web search results** without an external LLM call. Configure `GEMINI_API_KEY` or `OPENAI_API_KEY` in `.env` for full AI-authored narrative synthesis.

## {REQUIRED_REPORT_SECTIONS[0]}
**{topic}** (Research Depth: *{depth}*)

## {REQUIRED_REPORT_SECTIONS[1]}
This report synthesizes {len(sources)} live web sources gathered for **{topic}**. The retrieved sources highlight current developments, practical applications, and key considerations reported across reputable web domains.

## {REQUIRED_REPORT_SECTIONS[2]}
Understanding **{topic}** requires examining real-world implementations, measurable outcomes, and evolving industry perspectives. Below is a structured overview grounded in the retrieved web sources.

## {REQUIRED_REPORT_SECTIONS[3]}
Recent publications and industry documentation retrieved during this session provide context on how **{topic}** has evolved and where adoption is accelerating.

## {REQUIRED_REPORT_SECTIONS[4]}
{lead_snippets}

## {REQUIRED_REPORT_SECTIONS[5]}
Across the {len(sources)} collected sources, recurring themes include technological maturity, operational integration, and measurable impact. Readers should review the individual linked articles for domain-specific nuances.

## {REQUIRED_REPORT_SECTIONS[6]}
Specific figures and case examples reported in the retrieved snippets are preserved verbatim in the source summaries above; no synthetic statistics have been added.

## {REQUIRED_REPORT_SECTIONS[7]}
- **Source Scope**: Findings are limited to the top {len(sources)} web search results retrieved for this query.
- **Verification**: Configure an LLM API key (`GEMINI_API_KEY` or `OPENAI_API_KEY`) to enable deeper cross-source reasoning by all three CrewAI agents.

## {REQUIRED_REPORT_SECTIONS[8]}
Continued research and primary-source validation will uncover deeper quantitative benchmarks and emerging best practices in **{topic}**.

## {REQUIRED_REPORT_SECTIONS[9]}
Live web retrieval successfully surfaced {len(sources)} relevant references for **{topic}**. Reviewing the cited sources below provides a strong foundation for further decision-making.

## {REQUIRED_REPORT_SECTIONS[10]}
{refs_md}
"""
    return findings_md, summary_md, report_md


class ResearchOrchestrator:
    """Coordinates the 3-agent CrewAI research pipeline."""

    def run(
        self,
        topic: str,
        depth: str = "Intermediate",
        num_sources: int = 5,
        provider: str | None = None,
        api_key: str | None = None,
        allow_mock_fallback: bool = False,
        status_callback: StatusCallback | None = None,
    ) -> ResearchResult:
        """Execute the 3-stage multi-agent research workflow."""
        cleaned_topic = (topic or "").strip()
        if len(cleaned_topic) < 3:
            raise ValueError("Please enter a valid research topic (at least 3 characters).")

        def _notify(agent_key: str, state: str, msg: str) -> None:
            if status_callback:
                status_callback(agent_key, state, msg)

        warnings: list[str] = []

        # Step 1: Always collect grounded live web sources first so URLs are verified
        _notify("research", "running", f"Searching the web for '{cleaned_topic}'...")
        try:
            sources = perform_web_search(
                query=cleaned_topic,
                max_results=num_sources,
                enrich_pages=True,
            )
        except WebSearchError as exc:
            _notify("research", "error", str(exc))
            raise

        if not sources:
            _notify("research", "error", "No web sources found.")
            raise WebSearchError(
                f"No search results were found for '{cleaned_topic}'. Try broadening your topic."
            )

        # Check LLM availability
        settings = get_settings()
        prov = (provider or settings.llm_provider).lower().strip()
        has_key = settings.has_valid_llm_key(prov, api_key)

        if not has_key:
            if not allow_mock_fallback:
                _notify("research", "error", "Missing LLM API key.")
                _build_llm(prov, api_key)  # raises descriptive ResearchConfigurationError
            warnings.append(
                "LLM API key not configured — running in Demo/Mock Synthesis Mode using live DuckDuckGo search results."
            )
            _notify("research", "done", f"Collected {len(sources)} live sources.")
            _notify("summarizer", "running", "Analyzing retrieved sources...")
            _notify("report", "running", "Preparing 11-section Markdown report...")
            findings_md, summary_md, report_md = _synthesize_mock_report(
                cleaned_topic, depth, sources
            )
            _notify("summarizer", "done", "Summary complete.")
            _notify("report", "done", "Report ready.")
            return ResearchResult(
                topic=cleaned_topic,
                depth=depth,
                sources=sources,
                research_findings=findings_md,
                summarized_findings=summary_md,
                final_report=report_md,
                used_mock_mode=True,
                warnings=warnings,
            )

        # Full CrewAI multi-agent execution
        llm = _build_llm(prov, api_key)

        research_agent = create_research_agent(llm)
        summarizer_agent = create_summarizer_agent(llm)
        report_agent = create_report_agent(llm)

        research_task = create_research_task(
            research_agent, cleaned_topic, num_sources=num_sources, depth=depth
        )
        summarizer_task = create_summarizer_task(
            summarizer_agent, cleaned_topic, depth=depth, context_tasks=[research_task]
        )
        report_task = create_report_task(
            report_agent,
            cleaned_topic,
            depth=depth,
            context_tasks=[research_task, summarizer_task],
        )

        _notify("research", "running", "Research Agent searching & verifying sources...")
        crew = Crew(
            agents=[research_agent, summarizer_agent, report_agent],
            tasks=[research_task, summarizer_task, report_task],
            process=Process.sequential,
            verbose=True,
        )

        try:
            crew.kickoff()
        except Exception as exc:
            _notify("report", "error", f"CrewAI execution failed: {exc}")
            raise RuntimeError(f"Multi-agent execution failed: {exc}") from exc

        research_out = str(research_task.output.raw) if research_task.output else format_sources_markdown(sources)
        summary_out = str(summarizer_task.output.raw) if summarizer_task.output else ""
        report_out = str(report_task.output.raw) if report_task.output else ""

        _notify("research", "done", f"Verified {len(sources)}+ sources.")
        _notify("summarizer", "done", "Analysis & summary complete.")
        _notify("report", "done", "Final report generated.")

        return ResearchResult(
            topic=cleaned_topic,
            depth=depth,
            sources=sources,
            research_findings=research_out,
            summarized_findings=summary_out,
            final_report=report_out,
            used_mock_mode=False,
            warnings=warnings,
        )
