# ResearchAI – Multi-Agent Intelligent Research Assistant

A complete multi-agent AI Research Assistant built with **CrewAI**, **Gradio**, **DuckDuckGo Search / Tavily**, **BeautifulSoup4**, and **Google Gemini / OpenAI** for the **Skill Nexis Generative AI & LLM Development (Week 3)** submission.

Enter any research topic and three specialized AI agents collaborate sequentially to search the live web, analyze and summarize findings, and generate a structured 11-section research report with verified clickable citations and one-click **Markdown** & **PDF** downloads.

---

## Multi-Agent Architecture

1. **Research Agent (`Senior Web Research Analyst`)**
   - Searches the web via DuckDuckGo (`ddgs`) or optional Tavily API, enriches snippets with `BeautifulSoup4`, and collects credible sources with titles, URLs, domains, and publication dates.
2. **Summarizer Agent (`Expert Research Summarizer`)**
   - Analyzes collected findings, removes duplicates, compares viewpoints, highlights verified statistics, and preserves all source URLs.
3. **Report Generator Agent (`Professional Research Report Writer`)**
   - Produces a structured Markdown report containing all **11 required sections**:
     1. Research Topic
     2. Executive Summary
     3. Introduction
     4. Background and Context
     5. Key Research Findings
     6. Detailed Analysis
     7. Important Statistics and Examples
     8. Challenges and Limitations
     9. Future Opportunities
     10. Conclusion
     11. References and Sources

---

## Project Structure

```text
research-ai/
├── app.py                     # Application entry point (launches Gradio UI)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment configuration template
├── .gitignore
├── README.md
├── agents/
│   ├── __init__.py
│   ├── research_agent.py      # Agent 1: Senior Web Research Analyst
│   ├── summarizer_agent.py    # Agent 2: Expert Research Summarizer
│   └── report_agent.py        # Agent 3: Professional Research Report Writer
├── tools/
│   ├── __init__.py
│   └── web_search.py          # DuckDuckGo + Tavily + BeautifulSoup4 search tool
├── workflow/
│   ├── __init__.py
│   └── orchestrator.py        # CrewAI sequential orchestration + status tracking
├── frontend/
│   ├── __init__.py
│   └── interface.py           # Gradio web dashboard + live agent status cards
├── utils/
│   ├── __init__.py
│   ├── config.py              # Pydantic settings (.env loader)
│   └── report_export.py       # Markdown & PDF report exporters
├── outputs/                   # Generated Markdown and PDF reports
└── tests/
    ├── test_agents.py         # Agent & web search unit tests
    └── test_workflow.py       # End-to-end 5-topic workflow & export tests
```

---

## Quick Start

### 1. Prerequisites
- Python **3.10 – 3.13** (CrewAI requires `<3.14`).

### 2. Navigate & Activate Virtual Environment
```powershell
cd research-ai
.\.venv313\Scripts\Activate.ps1
```
*(Or create your own venv with Python 3.11–3.13 and run `pip install -r requirements.txt`.)*

### 3. Configure API Keys
Copy `.env.example` to `.env`:
```powershell
copy .env.example .env
```
Add your **Google Gemini** or **OpenAI** API key to `.env`:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_actual_key_here
```
*(You can also leave the key blank and enable **Demo / Mock Synthesis Mode** in the UI to test the live DuckDuckGo web search and 11-section report export without an LLM key.)*

### 4. Run the Application
```powershell
python app.py
```
Open **http://127.0.0.1:7860** in your browser.

### 5. Run Automated Tests
```powershell
pytest tests/ -v
```
