"""Agent 1: Senior Web Research Analyst."""

from typing import Any
from crewai import Agent, Task

from tools.web_search import web_search_tool


def create_research_agent(llm: Any) -> Agent:
    """Create the Research Agent responsible for live web search and fact gathering."""
    return Agent(
        role="Senior Web Research Analyst",
        goal=(
            "Search the internet for reliable, up-to-date information on the given "
            "research topic, collect at least the requested number of credible sources, "
            "and record exact titles, URLs, facts, statistics, and examples."
        ),
        backstory=(
            "You are an experienced investigative research analyst who verifies every "
            "claim against real web sources. You never invent URLs, statistics, or facts, "
            "and you clearly distinguish verified data from commentary."
        ),
        tools=[web_search_tool],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_research_task(
    agent: Agent,
    topic: str,
    num_sources: int = 5,
    depth: str = "Intermediate",
) -> Task:
    """Create the CrewAI Task for gathering web research findings."""
    return Task(
        description=(
            f"Research the topic: '{topic}'.\n"
            f"Research depth: {depth}.\n"
            f"Use the Web Research Search Tool to find at least {num_sources} credible sources.\n"
            "Requirements:\n"
            "1. Collect important facts, key developments, statistics, and concrete examples.\n"
            "2. Record every source's exact Title, clickable URL, domain, and publication date (if available).\n"
            "3. Avoid irrelevant fluff and NEVER fabricate facts or URLs.\n"
            "4. If reliable information is limited, explicitly state what could and could not be verified."
        ),
        expected_output=(
            "A structured collection of verified research findings organized by source, "
            "including source titles, exact URLs, key facts, statistics, and examples."
        ),
        agent=agent,
    )
