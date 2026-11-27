"""Modern Gradio web dashboard for ResearchAI – Intelligent Research Assistant."""

from pathlib import Path
import sys
from typing import Iterator

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.web_search import SearchSource, format_sources_markdown  # noqa: E402
from utils.config import get_settings  # noqa: E402
from utils.report_export import export_markdown, export_pdf  # noqa: E402
from workflow.orchestrator import (  # noqa: E402
    ResearchConfigurationError,
    ResearchOrchestrator,
)


CUSTOM_CSS = """
.hero-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 55%, #1d4ed8 100%);
    border-radius: 14px;
    padding: 22px 28px;
    color: #f8fafc;
    margin-bottom: 16px;
    box-shadow: 0 8px 24px rgba(30, 27, 75, 0.25);
}
.hero-banner h1 {
    margin: 0 0 6px 0;
    font-size: 1.75rem;
    font-weight: 700;
    color: #ffffff;
}
.hero-banner p {
    margin: 0;
    color: #e0e7ff;
    font-size: 0.96rem;
}
.status-card {
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 12px 14px;
    background: #f8fafc;
    color: #0f172a;
}
.status-running {
    border-left: 4px solid #3b82f6;
    background: #eff6ff;
}
.status-done {
    border-left: 4px solid #10b981;
    background: #ecfdf5;
}
.status-error {
    border-left: 4px solid #ef4444;
    background: #fef2f2;
}
.status-idle {
    border-left: 4px solid #94a3b8;
}
"""


def _status_html(title: str, subtitle: str, state: str, detail: str) -> str:
    icon_map = {
        "idle": "⏳",
        "running": "🔄",
        "done": "✅",
        "error": "❌",
    }
    cls = f"status-card status-{state}"
    icon = icon_map.get(state, "⏳")
    return f"""
    <div class="{cls}">
        <div style="font-weight:700; font-size:0.95rem;">{icon} {title}</div>
        <div style="font-size:0.82rem; color:#475569; margin-top:2px;">{subtitle}</div>
        <div style="font-size:0.8rem; color:#1e293b; margin-top:6px;">{detail}</div>
    </div>
    """


def _render_statuses(states: dict[str, tuple[str, str]]) -> tuple[str, str, str]:
    r_state, r_msg = states["research"]
    s_state, s_msg = states["summarizer"]
    g_state, g_msg = states["report"]
    return (
        _status_html("1. Research Agent", "Searching the web", r_state, r_msg),
        _status_html("2. Summarizer Agent", "Analyzing information", s_state, s_msg),
        _status_html("3. Report Generator Agent", "Preparing report", g_state, g_msg),
    )


def _format_citations_tab(sources: list[SearchSource]) -> str:
    if not sources:
        return "*No source citations available yet.*"
    return "### Verified Web Sources Used\n\n" + format_sources_markdown(sources)


def run_research_pipeline(
    topic: str,
    depth: str,
    num_sources: int,
    provider: str,
    api_key: str,
    demo_mode: bool,
) -> Iterator[tuple]:
    """Generator callback for Gradio that streams agent status updates and final outputs."""
    states = {
        "research": ("idle", "Waiting to start"),
        "summarizer": ("idle", "Waiting for Research Agent"),
        "report": ("idle", "Waiting for Summarizer Agent"),
    }

    cleaned_topic = (topic or "").strip()
    if len(cleaned_topic) < 3:
        states["research"] = ("error", "Invalid topic")
        s1, s2, s3 = _render_statuses(states)
        yield (
            s1,
            s2,
            s3,
            "⚠️ **Error:** Please enter a valid research topic (at least 3 characters).",
            "",
            "",
            "",
            None,
            None,
        )
        return

    # Emit initial running state
    states["research"] = ("running", f"Searching web for '{cleaned_topic}'...")
    s1, s2, s3 = _render_statuses(states)
    yield (s1, s2, s3, "🔄 Running multi-agent research pipeline...", "", "", "", None, None)

    orchestrator = ResearchOrchestrator()
    settings = get_settings()

    def _on_status(agent_key: str, state: str, msg: str) -> None:
        states[agent_key] = (state, msg)

    try:
        result = orchestrator.run(
            topic=cleaned_topic,
            depth=depth,
            num_sources=int(num_sources),
            provider=provider,
            api_key=api_key.strip() or None,
            allow_mock_fallback=demo_mode,
            status_callback=_on_status,
        )
    except ResearchConfigurationError as exc:
        states["research"] = ("error", "Missing API key")
        s1, s2, s3 = _render_statuses(states)
        yield (
            s1,
            s2,
            s3,
            f"🔑 **Configuration Required:** {exc}",
            "",
            "",
            "",
            None,
            None,
        )
        return
    except Exception as exc:
        s1, s2, s3 = _render_statuses(states)
        yield (
            s1,
            s2,
            s3,
            f"❌ **Research Failed:** {exc}",
            "",
            "",
            "",
            None,
            None,
        )
        return

    # Export Markdown & PDF
    md_path = export_markdown(result.final_report, cleaned_topic, settings.output_path)
    pdf_path = export_pdf(result.final_report, cleaned_topic, settings.output_path)

    s1, s2, s3 = _render_statuses(states)
    citations_md = _format_citations_tab(result.sources)

    yield (
        s1,
        s2,
        s3,
        result.final_report,
        result.research_findings,
        result.summarized_findings,
        citations_md,
        str(md_path),
        str(pdf_path),
    )


def clear_history() -> tuple:
    """Reset all status indicators and output panels."""
    states = {
        "research": ("idle", "Waiting to start"),
        "summarizer": ("idle", "Waiting for Research Agent"),
        "report": ("idle", "Waiting for Summarizer Agent"),
    }
    s1, s2, s3 = _render_statuses(states)
    return ("", s1, s2, s3, "", "", "", "", None, None)


def build_interface() -> gr.Blocks:
    """Build and return the ResearchAI Gradio Blocks dashboard."""
    settings = get_settings()

    with gr.Blocks(
        title="ResearchAI – Intelligent Research Assistant",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css=CUSTOM_CSS,
    ) as demo:
        gr.HTML(
            """
            <div class="hero-banner">
                <h1>🔬 ResearchAI – Intelligent Research Assistant</h1>
                <p>
                    Multi-Agent Research System powered by CrewAI • Live Web Search •
                    Structured 11-Section Reports with Verified Citations
                </p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                topic_input = gr.Textbox(
                    label="Research Topic",
                    placeholder="e.g., Artificial Intelligence in Healthcare",
                    lines=2,
                )
                gr.Examples(
                    examples=[
                        ["Artificial Intelligence in Healthcare"],
                        ["Generative AI and Large Language Models"],
                        ["Climate Change and Renewable Energy"],
                        ["Cybersecurity and Data Privacy"],
                        ["Future of Autonomous Vehicles"],
                    ],
                    inputs=topic_input,
                    label="Sample Research Topics",
                )

            with gr.Column(scale=1):
                depth_input = gr.Radio(
                    choices=["Basic", "Intermediate", "Detailed"],
                    value=settings.default_depth,
                    label="Research Depth",
                )
                num_sources_input = gr.Slider(
                    minimum=3,
                    maximum=10,
                    step=1,
                    value=settings.default_num_sources,
                    label="Number of Sources",
                )

        with gr.Accordion("⚙️ LLM & Mode Settings", open=False):
            with gr.Row():
                provider_input = gr.Dropdown(
                    choices=["gemini", "openai"],
                    value=settings.llm_provider,
                    label="LLM Provider",
                )
                api_key_input = gr.Textbox(
                    label="API Key (optional override)",
                    type="password",
                    placeholder="Uses .env key if left blank",
                )
                demo_mode_input = gr.Checkbox(
                    label="Demo / Mock Synthesis Mode (when no API key is set)",
                    value=not settings.has_valid_llm_key(),
                )

        with gr.Row():
            run_btn = gr.Button("🚀 Generate Research Report", variant="primary", scale=3)
            clear_btn = gr.Button("🧹 Clear Research History", variant="secondary", scale=1)

        gr.Markdown("### 🤖 Multi-Agent Pipeline Status")
        with gr.Row():
            status_research = gr.HTML(
                _status_html("1. Research Agent", "Searching the web", "idle", "Waiting to start")
            )
            status_summarizer = gr.HTML(
                _status_html("2. Summarizer Agent", "Analyzing information", "idle", "Waiting for Research Agent")
            )
            status_report = gr.HTML(
                _status_html("3. Report Generator Agent", "Preparing report", "idle", "Waiting for Summarizer Agent")
            )

        with gr.Tabs():
            with gr.Tab("📄 Final Structured Report"):
                final_report_md = gr.Markdown(
                    value="*Enter a topic above and click **Generate Research Report**.*"
                )
            with gr.Tab("🔍 Research Findings"):
                research_findings_md = gr.Markdown()
            with gr.Tab("📝 Summarized Findings"):
                summarized_findings_md = gr.Markdown()
            with gr.Tab("🔗 Source Citations"):
                citations_md = gr.Markdown()

        with gr.Row():
            md_download = gr.File(label="⬇️ Download Report (Markdown)", interactive=False)
            pdf_download = gr.File(label="⬇️ Download Report (PDF)", interactive=False)

        run_btn.click(
            fn=run_research_pipeline,
            inputs=[
                topic_input,
                depth_input,
                num_sources_input,
                provider_input,
                api_key_input,
                demo_mode_input,
            ],
            outputs=[
                status_research,
                status_summarizer,
                status_report,
                final_report_md,
                research_findings_md,
                summarized_findings_md,
                citations_md,
                md_download,
                pdf_download,
            ],
        )

        clear_btn.click(
            fn=clear_history,
            inputs=[],
            outputs=[
                topic_input,
                status_research,
                status_summarizer,
                status_report,
                final_report_md,
                research_findings_md,
                summarized_findings_md,
                citations_md,
                md_download,
                pdf_download,
            ],
        )

    return demo
