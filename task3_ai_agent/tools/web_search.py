"""Web Search Tool supporting Tavily API and DuckDuckGo search fallback.

Features:
- Prioritizes Tavily API when TAVILY_API_KEY is configured in the environment.
- Automatically falls back to DuckDuckGo search when Tavily key is absent or exhausted.
- Sanitizes untrusted web snippets to defend against prompt injection attacks.
- Gracefully handles rate limits, offline states, and timeouts without crashing the agent.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Any, Dict, List, Optional
from task3_ai_agent.tools.base import BaseAgentTool, ToolResult, WebSearchInput

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*duckduckgo_search.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*duckduckgo_search.*")

MAX_SNIPPET_LENGTH = 350


def sanitize_web_snippet(text: str) -> str:
    """Sanitizes raw web text to neutralize prompt injection patterns."""
    if not text:
        return ""
    # Strip dangerous directives that attempt to override system instructions
    cleaned = re.sub(r"(?i)(ignore (all )?previous instructions|system prompt:|you are now)", "[REDACTED_DIRECTIVE]", text)
    # Remove control characters
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    # Normalize excessive spaces and linebreaks
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:MAX_SNIPPET_LENGTH]


class WebSearchTool(BaseAgentTool):
    """Tool that queries the web via Tavily or DuckDuckGo search."""

    name = "web_search"
    description = (
        "Searches the web for up-to-date facts, statistics, recent news, or general world knowledge. "
        "Input should be a clear search query string (e.g. 'Python latest version 2026', 'Datheon AI solutions'). "
        "Returns top search result snippets with URLs."
    )
    args_schema = WebSearchInput

    def __init__(self, tavily_api_key: Optional[str] = None):
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")

    def _search_tavily(self, query: str, max_results: int) -> List[Dict[str, str]]:
        from tavily import TavilyClient
        client = TavilyClient(api_key=self.tavily_api_key)
        response = client.search(query=query, max_results=max_results)
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "snippet": sanitize_web_snippet(r.get("content", "")),
                "url": r.get("url", ""),
            })
        return results

    def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict[str, str]]:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                raw_items = list(ddgs.text(query, max_results=max_results))
                for item in raw_items:
                    results.append({
                        "title": item.get("title", ""),
                        "snippet": sanitize_web_snippet(item.get("body", "")),
                        "url": item.get("href", ""),
                    })
        return results

    def _search_duckduckgo_instant(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Official DuckDuckGo Instant Answer REST API (rate-limit immune)."""
        import json
        import urllib.parse
        import urllib.request
        q_enc = urllib.parse.quote(query)
        url = f"https://api.duckduckgo.com/?q={q_enc}&format=json&no_html=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        results = []
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                abstract = data.get("Abstract")
                if abstract:
                    results.append({
                        "title": data.get("Heading") or query,
                        "snippet": sanitize_web_snippet(abstract),
                        "url": data.get("AbstractURL") or "https://duckduckgo.com",
                    })
                for topic in data.get("RelatedTopics", []):
                    if len(results) >= max_results:
                        break
                    if "Text" in topic:
                        results.append({
                            "title": topic.get("Text", "")[:60],
                            "snippet": sanitize_web_snippet(topic.get("Text", "")),
                            "url": topic.get("FirstURL", ""),
                        })
        except Exception:
            pass
        return results

    def _search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Official Wikipedia Search REST API fallback for factual and encyclopedic inquiries."""
        import json
        import urllib.parse
        import urllib.request
        q_enc = urllib.parse.quote(query)
        url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q_enc}&format=json&srlimit={max_results}"
        req = urllib.request.Request(url, headers={"User-Agent": "DatheonInternshipAgent/1.0 (edu; Python3.13)"})
        results = []
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
                items = data.get("query", {}).get("search", [])
                for idx, item in enumerate(items):
                    raw_snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                    clean_snippet = sanitize_web_snippet(raw_snippet)
                    title = item.get("title", "")
                    if idx == 0 and title:
                        try:
                            t_enc = urllib.parse.quote(title.replace(" ", "_"))
                            intro_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{t_enc}"
                            intro_req = urllib.request.Request(intro_url, headers={"User-Agent": "DatheonInternshipAgent/1.0"})
                            with urllib.request.urlopen(intro_req, timeout=4) as i_resp:
                                i_data = json.loads(i_resp.read().decode())
                                extract = i_data.get("extract", "")
                                if extract:
                                    clean_snippet = sanitize_web_snippet(extract)
                        except Exception:
                            pass
                    results.append({
                        "title": title,
                        "snippet": clean_snippet,
                        "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}",
                    })
        except Exception:
            pass
        return results

    def run(self, query: str, max_results: int = 3) -> ToolResult:
        clean_query = str(query).strip(" \t\n\r\"'`")
        if "query" in clean_query and ("{" in clean_query or ":" in clean_query):
            try:
                import json
                data = json.loads(clean_query)
                if isinstance(data, dict) and "query" in data:
                    clean_query = str(data["query"]).strip(" \t\n\r\"'`")
            except Exception:
                import re
                match = re.search(r"['\"]query['\"]\s*:\s*['\"]([^'\"\}\n\r`]+)['\"]", clean_query)
                if match:
                    clean_query = match.group(1).strip(" \t\n\r\"'`")

        if not clean_query:
            return ToolResult(
                success=False,
                output="",
                error_message="Search query cannot be empty.",
            )

        provider_used = "none"
        items: List[Dict[str, str]] = []
        last_error = None

        # 1. Try Tavily if key is available
        if self.tavily_api_key and "your_tavily" not in self.tavily_api_key:
            try:
                items = self._search_tavily(clean_query, max_results)
                if items:
                    provider_used = "tavily"
            except Exception as exc:
                last_error = exc

        # 2. Try DuckDuckGo scraper fallback
        if not items:
            try:
                items = self._search_duckduckgo(clean_query, max_results)
                if items:
                    provider_used = "duckduckgo"
            except Exception as exc:
                last_error = exc

        # 3. Try DuckDuckGo Instant Answer REST API
        if not items:
            try:
                items = self._search_duckduckgo_instant(clean_query, max_results)
                if items:
                    provider_used = "duckduckgo_instant"
            except Exception as exc:
                last_error = exc

        # 4. Try Wikipedia Search REST API fallback
        if not items:
            try:
                items = self._search_wikipedia(clean_query, max_results)
                if items:
                    provider_used = "wikipedia"
            except Exception as exc:
                last_error = exc

        if not items:
            err_detail = f" ({str(last_error)})" if last_error else ""
            return ToolResult(
                success=False,
                output="",
                error_message=(
                    f"Web search service is currently unavailable or returned 0 results{err_detail}. "
                    "Please check network connection or try alternative keywords."
                ),
            )

        # Format output string
        formatted_parts = [f"Web Search Results for '{clean_query}':"]
        for idx, it in enumerate(items, 1):
            formatted_parts.append(
                f"[{idx}] {it['title']}\n"
                f"    Snippet: {it['snippet']}\n"
                f"    Source URL: {it['url']}"
            )

        output_str = "\n".join(formatted_parts)
        return ToolResult(
            success=True,
            output=output_str,
            metadata={
                "provider": provider_used,
                "query": clean_query,
                "result_count": len(items),
            },
        )

