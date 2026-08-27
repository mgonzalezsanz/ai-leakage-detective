"""Local RAG over the /data/knowledge_base policy + internal-notes corpus.

Embeddings run fully offline via sentence-transformers; the FAISS index is built once, in memory, lazily on first search.
"""

import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

KB_DIR = Path(__file__).parent.parent / "data" / "knowledge_base"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Lazy singleton: stay None/empty until the first search() call, then built once and kept resident in 
# memory for the life of this process - no reload, no explicit unload.
# Each process (Streamlit, langgraph dev, a python -m agent.evals run) holds its own copy.
_model: SentenceTransformer | None = None
_index: faiss.Index | None = None
_chunks: list[dict] = []


def _split_into_chunks(path: Path) -> list[dict]:
    """Split a markdown file into one chunk per '## ' section (the whole
    file, minus its title, if it has none)."""
    text = path.read_text(encoding="utf-8")
    sections = re.split(r"\n(?=## )", text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section or section.startswith("# "):
            continue
        title = section.splitlines()[0].lstrip("#").strip()
        chunks.append({"source": str(path.relative_to(KB_DIR)), "title": title, "text": section})
    return chunks


def _build_index() -> None:
    global _model, _index, _chunks
    _model = SentenceTransformer(EMBEDDING_MODEL)
    _chunks = [chunk for path in sorted(KB_DIR.rglob("*.md")) for chunk in _split_into_chunks(path)]
    # Sentence-transformer models are trained/evaluated for cosine similarity, not raw distance - magnitude
    # varies for reasons unrelated to meaning (e.g. text length).
    # FAISS has no native cosine metric, so normalize to unit length + use inner product below: for unit 
    # vectors, dot(a, b) = |a| * |b| * cos(angle) = cos(angle), i.e. inner product is cosine similarity.
    embeddings = _model.encode([c["text"] for c in _chunks], normalize_embeddings=True)
    # Flat = exact (no approximation) search; fine at this corpus size. IP (inner product) over the normalized
    # vectors above gives cosine similarity - "higher = more similar"
    _index = faiss.IndexFlatIP(embeddings.shape[1])
    _index.add(np.asarray(embeddings, dtype="float32"))


def search(query: str, k: int = 3) -> list[dict]:
    """Return the top-k most relevant knowledge-base chunks for a query,
    each with its source file, section title, text, and similarity score."""
    if _index is None:
        _build_index()
    query_vec = _model.encode([query], normalize_embeddings=True)
    scores, idxs = _index.search(np.asarray(query_vec, dtype="float32"), min(k, len(_chunks)))
    return [
        {**_chunks[i], "score": round(float(score), 4)} for score, i in zip(scores[0], idxs[0]) if i != -1
    ]