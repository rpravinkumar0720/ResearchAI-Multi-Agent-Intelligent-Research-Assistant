"""End-to-end workflow and export tests for the 5 required research topics."""

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.report_agent import REQUIRED_REPORT_SECTIONS  # noqa: E402
from utils.report_export import export_markdown, export_pdf  # noqa: E402
from workflow.orchestrator import (  # noqa: E402
    ResearchConfigurationError,
    ResearchOrchestrator,
)


TEST_TOPICS = [
    "Artificial Intelligence in Healthcare",
    "Generative AI and Large Language Models",
    "Climate Change and Renewable Energy",
    "Cybersecurity and Data Privacy",
    "Future of Autonomous Vehicles",
]


def test_orchestrator_rejects_invalid_topic() -> None:
    orch = ResearchOrchestrator()
    with pytest.raises(ValueError):
        orch.run("", allow_mock_fallback=True)


def test_orchestrator_raises_when_key_missing_and_mock_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from utils.config import get_settings

    get_settings.cache_clear()
    orch = ResearchOrchestrator()
    with pytest.raises(ResearchConfigurationError):
        orch.run("Artificial Intelligence in Healthcare", allow_mock_fallback=False)
    get_settings.cache_clear()


@pytest.mark.parametrize("topic", TEST_TOPICS)
def test_end_to_end_five_topics_with_live_search_and_exports(
    topic: str, tmp_path: Path
) -> None:
    """Test all 5 required topics end-to-end with live web search, 11 sections, MD & PDF export."""
    orch = ResearchOrchestrator()
    statuses: list[tuple[str, str]] = []

    result = orch.run(
        topic=topic,
        depth="Intermediate",
        num_sources=5,
        allow_mock_fallback=True,
        status_callback=lambda agent, state, _msg: statuses.append((agent, state)),
    )

    assert len(result.sources) >= 3
    assert result.research_findings
    assert result.summarized_findings
    assert result.final_report

    for section in REQUIRED_REPORT_SECTIONS:
        assert section in result.final_report

    # Verify clickable source URLs appear in report
    assert any(src.url in result.final_report for src in result.sources)
    assert ("research", "done") in statuses
    assert ("summarizer", "done") in statuses
    assert ("report", "done") in statuses

    md_file = export_markdown(result.final_report, topic, tmp_path)
    pdf_file = export_pdf(result.final_report, topic, tmp_path)
    assert md_file.exists() and md_file.stat().st_size > 200
    assert pdf_file.exists() and pdf_file.stat().st_size > 500
