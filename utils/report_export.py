"""Markdown and PDF report export utilities."""

from datetime import datetime
from pathlib import Path
import re

from fpdf import FPDF


def _slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug[:60] or "research_report"


def _latin1_safe(text: str) -> str:
    """Normalize common Unicode punctuation so built-in PDF fonts render cleanly."""
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "*",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def export_markdown(report_md: str, topic: str, output_dir: Path) -> Path:
    """Save the research report as a `.md` file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_slugify(topic)}_{timestamp}.md"
    path = output_dir / filename
    path.write_text(report_md, encoding="utf-8")
    return path


def export_pdf(report_md: str, topic: str, output_dir: Path) -> Path:
    """Convert the Markdown research report to a formatted PDF and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{_slugify(topic)}_{timestamp}.pdf"
    path = output_dir / filename

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title banner
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(pdf.epw, 9, _latin1_safe(f"ResearchAI Report: {topic}"), wrapmode="CHAR")
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "I", 9)
    pdf.cell(
        pdf.epw,
        6,
        _latin1_safe(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
        new_x="LMARGIN",
        new_y="NEXT",
    )
    pdf.ln(4)

    for raw_line in report_md.splitlines():
        line = raw_line.strip()
        if not line:
            pdf.ln(3)
            continue

        # Strip markdown bold/code markers for clean PDF text
        clean_line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        clean_line = re.sub(r"`(.*?)`", r"\1", clean_line)
        clean_line = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", clean_line)

        pdf.set_x(pdf.l_margin)
        if clean_line.startswith("# "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(pdf.epw, 8, _latin1_safe(clean_line[2:]), wrapmode="CHAR")
        elif clean_line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(pdf.epw, 7, _latin1_safe(clean_line[3:]), wrapmode="CHAR")
        elif clean_line.startswith("### "):
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(pdf.epw, 6, _latin1_safe(clean_line[4:]), wrapmode="CHAR")
        elif clean_line.startswith(("- ", "* ")):
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(pdf.epw, 6, _latin1_safe(f"  * {clean_line[2:]}"), wrapmode="CHAR")
        else:
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(pdf.epw, 6, _latin1_safe(clean_line), wrapmode="CHAR")

    pdf.output(str(path))
    return path
