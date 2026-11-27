"""Agent 2: Expert Research Summarizer."""

from typing import Any
from crewai import Agent, Task


def create_summarizer_agent(llm: Any) -> Agent:
    """Create the Summarizer Agent responsible for synthesizing and deduplicating findings."""
    return Agent(
        role="Expert Research Summarizer",
        goal=(
            "Analyze raw web research findings, remove duplicate information, compare "
            "viewpoints, identify common themes, and produce a concise, structured summary "
            "while preserving all verified source URLs."
        ),
        backstory=(
            "You are a senior research editor skilled at distilling complex multi-source "
            "information into clear thematic insights without losing factual accuracy or citations."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_summarizer_task(
    agent: Agent,
    topic: str,
    depth: str,
    context_tasks: list[Task],
) -> Task:
    """Create the CrewAI Task for analyzing and summarizing research findings."""
    return Task(
        description=(
            f"Analyze the research findings gathered for '{topic}' (Depth: {depth}).\n"
            "Requirements:\n"
            "1. Identify the most important findings and common themes.\n"
            "2. Remove duplicate or redundant points.\n"
            "3. Compare differing viewpoints or trade-offs where present.\n"
            "4. Highlight verified statistics, metrics, and real-world examples.\n"
            "5. Distinguish verified facts from opinions.\n"
            "6. Preserve the exact source URLs alongside the findings they support."
        ),
        expected_output=(
            "A clear, well-organized thematic research summary with key facts, "
            "statistics, viewpoints, and preserved source URLs."
        ),
        agent=agent,
        context=context_tasks,
    )
