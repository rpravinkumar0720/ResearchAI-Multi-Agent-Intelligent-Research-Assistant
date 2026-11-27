"""Agent 3: Professional Research Report Writer."""

from typing import Any
from crewai import Agent, Task


REQUIRED_REPORT_SECTIONS = [
    "1. Research Topic",
    "2. Executive Summary",
    "3. Introduction",
    "4. Background and Context",
    "5. Key Research Findings",
    "6. Detailed Analysis",
    "7. Important Statistics and Examples",
    "8. Challenges and Limitations",
    "9. Future Opportunities",
    "10. Conclusion",
    "11. References and Sources",
]


def create_report_agent(llm: Any) -> Agent:
    """Create the Report Generator Agent responsible for writing the final Markdown report."""
    return Agent(
        role="Professional Research Report Writer",
        goal=(
            "Transform the synthesized research summary into a polished, publication-ready "
            "Markdown research report containing all 11 required sections and accurate "
            "clickable source citations."
        ),
        backstory=(
            "You are an executive technical writer who produces clear, rigorous research "
            "reports for industry leaders and academic reviewers. You strictly cite only "
            "the verified URLs provided in the research summary."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_report_task(
    agent: Agent,
    topic: str,
    depth: str,
    context_tasks: list[Task],
) -> Task:
    """Create the CrewAI Task for generating the 11-section Markdown report."""
    sections_list = "\n".join(f"- ## {s}" for s in REQUIRED_REPORT_SECTIONS)
    return Task(
        description=(
            f"Write a comprehensive, professional Markdown research report on '{topic}' "
            f"(Target Depth: {depth}).\n"
            "You MUST include all 11 sections below in exact order using `##` headings:\n"
            f"{sections_list}\n\n"
            "Strict Guidelines:\n"
            "1. Include inline citations (e.g., [Source Title](URL)) alongside key findings.\n"
            "2. Do NOT invent references, statistics, or URLs.\n"
            "3. Distinguish verified facts from analyst/industry opinions.\n"
            "4. In Section 11 (References and Sources), list every source used with clickable Markdown links."
        ),
        expected_output=(
            "A complete, professionally formatted Markdown research report containing "
            "all 11 required sections and clickable source URLs."
        ),
        agent=agent,
        context=context_tasks,
    )
