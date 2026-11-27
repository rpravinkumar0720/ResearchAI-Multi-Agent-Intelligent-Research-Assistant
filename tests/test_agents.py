"""Unit tests for CrewAI agents and web search tool."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.report_agent import (  # noqa: E402
    REQUIRED_REPORT_SECTIONS,
    create_report_agent,
    create_report_task,
)
from agents.research_agent import create_research_agent, create_research_task  # noqa: E402
from agents.summarizer_agent import (  # noqa: E402
    create_summarizer_agent,
    create_summarizer_task,
)
from tools.web_search import WebSearchError, perform_web_search  # noqa: E402


def test_agent_roles_and_tasks() -> None:
    """Verify all three CrewAI agents and tasks initialize with proper roles and context."""
    dummy_llm = "gpt-4o-mini"
    r_agent = create_research_agent(dummy_llm)
    s_agent = create_summarizer_agent(dummy_llm)
    g_agent = create_report_agent(dummy_llm)

    assert r_agent.role == "Senior Web Research Analyst"
    assert s_agent.role == "Expert Research Summarizer"
    assert g_agent.role == "Professional Research Report Writer"
    assert len(r_agent.tools) == 1

    r_task = create_research_task(r_agent, "Artificial Intelligence in Healthcare", num_sources=5)
    s_task = create_summarizer_task(s_agent, "Artificial Intelligence in Healthcare", "Intermediate", [r_task])
    g_task = create_report_task(g_agent, "Artificial Intelligence in Healthcare", "Intermediate", [r_task, s_task])

    assert "Artificial Intelligence in Healthcare" in r_task.description
    assert s_task.context == [r_task]
    assert g_task.context == [r_task, s_task]
    assert len(REQUIRED_REPORT_SECTIONS) == 11


def test_web_search_rejects_short_query() -> None:
    with pytest.raises(WebSearchError):
        perform_web_search("ai", max_results=3)


def test_live_web_search_returns_sources() -> None:
    """Verify real DuckDuckGo search retrieves valid URLs and snippets."""
    sources = perform_web_search("Artificial Intelligence in Healthcare", max_results=5)
    assert len(sources) >= 3
    for src in sources:
        assert src.url.startswith("http")
        assert src.title
        assert src.domain
