"""Reranking via the Cohere Rerank API, used to re-score FAISS-retrieved
knowledge-base candidates by actual relevance before confidence-labeling
them, rather than trusting raw embedding cosine similarity alone."""

import os

import httpx

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"
RERANK_MODEL = "rerank-v3.5"
REQUEST_TIMEOUT_SECONDS = 10


def rerank(query: str, documents: list[str]) -> list[float] | None:
    """Score each of `documents` for relevance to `query` via Cohere Rerank,
    returning one relevance score (0-1, higher = more relevant) per document,
    in the SAME order as `documents` was given - not Cohere's own ranked
    order. Returns None (not a list of zeros) if COHERE_API_KEY isn't set or
    the request fails, so callers can tell "not reranked" apart from "every
    document scored 0" and fall back to their own ranking instead."""
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key or not documents:
        return None
    try:
        response = httpx.post(
            COHERE_RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": RERANK_MODEL, "query": query, "documents": documents, "top_n": len(documents)},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError:
        return None

    scores = [0.0] * len(documents)
    for r in response.json().get("results", []):
        scores[r["index"]] = r["relevance_score"]
    return scores
