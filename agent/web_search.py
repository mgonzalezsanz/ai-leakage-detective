"""Web search fallback via the Tavily API, used when the local knowledge base
returns nothing or only low-confidence matches."""

import os

import httpx

TAVILY_SEARCH_URL = "https://api.tavily.com/search"
REQUEST_TIMEOUT_SECONDS = 10


def search(query: str, k: int = 3) -> list[dict]:
    """Run a real web search via Tavily and return up to k results, each with
    its source URL, title, text snippet, Tavily's relevance score, and
    source_type="web" (so callers can distinguish these from internal KB
    results without inspecting anything else). Returns a single-item list
    with an "error" key if TAVILY_API_KEY isn't set or the request fails."""
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return [{"error": "TAVILY_API_KEY not set"}]
    try:
        response = httpx.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": k, "search_depth": "basic"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        return [{"error": f"web search failed: {e}"}]

    return [
        {
            "source": r.get("url"),
            "title": r.get("title"),
            "text": r.get("content"),
            "score": r.get("score"),
            "source_type": "web",
        }
        for r in response.json().get("results", [])[:k]
    ]
