"""Web search tool supporting DuckDuckGo Search and optional Tavily API."""

from typing import Any
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from crewai.tools import tool
from pydantic import BaseModel, Field
import requests

from utils.config import get_settings


class WebSearchError(Exception):
    """Raised when web search fails or returns unusable results."""


class SearchSource(BaseModel):
    """Structured representation of a single web search result."""

    title: str
    url: str
    domain: str
    snippet: str
    published_date: str = "Not specified"


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "") or "unknown"
    except Exception:
        return "unknown"


def fetch_page_excerpt(url: str, timeout: int = 6, max_chars: int = 800) -> str:
    """Fetch a short clean text excerpt from a web page using BeautifulSoup4."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        if not resp.ok or "text/html" not in resp.headers.get("content-type", ""):
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        paragraphs = [
            p.get_text(" ", strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 40
        ]
        joined = " ".join(paragraphs[:4]).strip()
        return joined[:max_chars]
    except Exception:
        return ""


def search_duckduckgo(query: str, max_results: int = 5) -> list[SearchSource]:
    """Search the web using DuckDuckGo (`ddgs`) without requiring an API key."""
    import time
    from ddgs import DDGS

    sources: list[SearchSource] = []
    seen_urls: set[str] = set()
    fetch_count = max(max_results * 2, 10)

    results: list[dict[str, Any]] = []
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            results = DDGS().text(query, max_results=fetch_count) or []
            if results:
                break
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (attempt + 1))

    if not results and last_exc is not None:
        return []

    for item in results:
        url = (item.get("href") or item.get("url") or "").strip()
        title = (item.get("title") or "Untitled Source").strip()
        snippet = (item.get("body") or item.get("snippet") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        sources.append(
            SearchSource(
                title=title,
                url=url,
                domain=_extract_domain(url),
                snippet=snippet,
                published_date=item.get("published", "Not specified") or "Not specified",
            )
        )
        if len(sources) >= max_results:
            break
    return sources


def search_wikipedia(query: str, max_results: int = 5) -> list[SearchSource]:
    """Supplement web research using the public Wikipedia Search API (no key required)."""
    try:
        resp = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
                "format": "json",
                "utf8": 1,
            },
            headers={"User-Agent": "ResearchAI/1.0 (Educational Multi-Agent Assistant)"},
            timeout=8,
        )
        if not resp.ok:
            return []
        hits = resp.json().get("query", {}).get("search", [])
    except Exception:
        return []

    sources: list[SearchSource] = []
    for item in hits:
        title = item.get("title", "").strip()
        if not title:
            continue
        slug = title.replace(" ", "_")
        url = f"https://en.wikipedia.org/wiki/{slug}"
        raw_snippet = item.get("snippet", "")
        clean_snippet = BeautifulSoup(raw_snippet, "html.parser").get_text(" ", strip=True)
        sources.append(
            SearchSource(
                title=f"{title} - Wikipedia",
                url=url,
                domain="en.wikipedia.org",
                snippet=clean_snippet,
                published_date=(item.get("timestamp") or "Not specified")[:10],
            )
        )
    return sources


def search_tavily(query: str, api_key: str, max_results: int = 5) -> list[SearchSource]:
    """Search the web using Tavily API when TAVILY_API_KEY is configured."""
    from tavily import TavilyClient

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
    except Exception as exc:
        raise WebSearchError(f"Tavily search failed: {exc}") from exc

    sources: list[SearchSource] = []
    for item in response.get("results", []):
        url = (item.get("url") or "").strip()
        if not url:
            continue
        sources.append(
            SearchSource(
                title=(item.get("title") or "Untitled Source").strip(),
                url=url,
                domain=_extract_domain(url),
                snippet=(item.get("content") or "").strip(),
                published_date=item.get("published_date") or "Not specified",
            )
        )
    return sources


def perform_web_search(
    query: str,
    max_results: int = 5,
    enrich_pages: bool = False,
) -> list[SearchSource]:
    """Perform live web search (Tavily if configured, otherwise DuckDuckGo).

    Validates query and handles empty or failed search responses gracefully.
    """
    cleaned = query.strip()
    if len(cleaned) < 3:
        raise WebSearchError("Research topic must be at least 3 characters long.")

    settings = get_settings()
    tavily_key = settings.tavily_api_key.strip()

    sources: list[SearchSource] = []
    if tavily_key:
        try:
            sources = search_tavily(cleaned, tavily_key, max_results=max_results)
        except WebSearchError:
            sources = search_duckduckgo(cleaned, max_results=max_results)
    else:
        sources = search_duckduckgo(cleaned, max_results=max_results)

    if len(sources) < max_results:
        seen_urls = {s.url for s in sources}
        for wiki_src in search_wikipedia(cleaned, max_results=max_results):
            if wiki_src.url not in seen_urls:
                sources.append(wiki_src)
                seen_urls.add(wiki_src.url)
            if len(sources) >= max_results:
                break

    if enrich_pages and sources:
        for src in sources[:3]:
            excerpt = fetch_page_excerpt(src.url)
            if excerpt and len(excerpt) > len(src.snippet):
                src.snippet = excerpt

    return sources


def format_sources_markdown(sources: list[SearchSource]) -> str:
    """Format collected SearchSource items into readable Markdown with clickable links."""
    if not sources:
        return "No web sources were retrieved."
    lines: list[str] = []
    for idx, src in enumerate(sources, start=1):
        lines.append(
            f"{idx}. **[{src.title}]({src.url})** (`{src.domain}`)\n"
            f"   - *Published*: {src.published_date}\n"
            f"   - *Summary*: {src.snippet}"
        )
    return "\n\n".join(lines)


class WebSearchInput(BaseModel):
    query: str = Field(..., description="The search query or research topic.")
    max_results: int = Field(default=5, description="Maximum number of web sources to fetch.")


@tool("Web Research Search Tool")
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Search the web for reliable articles, facts, statistics, and source URLs."""
    try:
        sources = perform_web_search(query=query, max_results=max_results, enrich_pages=True)
    except WebSearchError as exc:
        return f"Search error: {exc}"
    if not sources:
        return "No search results found for this query."
    return format_sources_markdown(sources)
