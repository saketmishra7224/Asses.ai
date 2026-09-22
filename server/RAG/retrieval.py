"""Candidate-scoped hybrid retrieval (Phase 3).

Given a candidate_id + natural-language query, returns the most relevant
chunks from that candidate's resume, GitHub and LeetCode knowledge:

  vector_results (KNN on query embedding, candidate TAG filter)
  + keyword_results (full-text terms, candidate TAG filter)
  ↓ Reciprocal Rank Fusion (transparent, debuggable)
  ↓ per-source diversity cap
  ↓ top-K SearchResults (origin="hybrid")

Isolation rule: every path filters on candidate_id at the query/index
level first (never global-search-then-filter); results are re-checked with
``check_candidates_match`` before return. An empty candidate or no matches
yields ``[]`` — never fabricated content.

Score convention (all origins): lower = more relevant.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple

from .embeddings import EmbeddingProvider, get_embedding_provider
from .index import RagConfig, VectorStore, get_vector_store
from .models import SearchFilters, SearchResult, check_candidates_match

RRF_K = 60
OVERSAMPLE_FACTOR = 2
OVERSAMPLE_MINIMUM = 10

STOPWORDS = frozenset("""
a an the and or but of at by for with about into through during before after
above below to from up down in out on off over under again further then once
here there when where why how all any both each few more most other some such
no nor not only own same so than too very can will just don should now is are
was were be been being have has had having do does did doing would could ought
i you he she it we they them his her its our their this that these those am me
my your his her its our their what which who whom whose tell describe explain
give list name talk walk through
""".split())


def extract_terms(query: str, min_len: int = 2, max_terms: int = 12) -> List[str]:
    """Split a question into lexical search terms.

    Keeps token characters useful for tech terms (``Next.js``, ``C++``);
    drops stopwords and short tokens. Returns [] when nothing searchable
    remains (caller then skips the keyword path).
    """
    tokens = re.findall(r"[A-Za-z0-9_+#.\-]+", query or "")
    seen, terms = set(), []
    for tok in tokens:
        low = tok.lower().strip(".-")  # keep C++/C# intact, drop sentence periods
        if len(low) < min_len or low in STOPWORDS or low in seen:
            continue
        seen.add(low)
        terms.append(low)
        if len(terms) >= max_terms:
            break
    return terms


def rrf_fuse(ranked_lists: List[List[SearchResult]], k: int = RRF_K) -> List[Tuple[SearchResult, float]]:
    """Reciprocal Rank Fusion over ranked lists; dedupes by chunk key.

    fused(doc) = Σ 1/(k + rank) with 1-based ranks. Returns
    [(SearchResult, fused_score)] sorted by fused_score descending.
    Higher fused_score = more relevant (converted to a distance score
    by the caller to keep the lower-is-better convention).
    """
    totals: Dict[str, float] = {}
    best: Dict[str, SearchResult] = {}
    for ranked in ranked_lists:
        for rank, res in enumerate(ranked, start=1):
            key = res.key or res.document.document_id
            totals[key] = totals.get(key, 0.0) + 1.0 / (k + rank)
            if key not in best:
                best[key] = res
    fused = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [(best[key], score) for key, score in fused]


def apply_diversity(
    ranked: List[SearchResult],
    top_k: int,
    max_per_source: int = 3,
) -> List[SearchResult]:
    """Greedy diversity cap: at most `max_per_source` chunks per source.

    Walks the ranked list in order (rank 1 always kept) so relevance still
    dominates, but one prolific source (e.g. 8 README windows) cannot crowd
    out resume/LeetCode evidence when relevance is comparable.
    """
    out, per_source = [], {}
    for res in ranked:
        n = per_source.get(res.document.source, 0)
        if not out or n < max_per_source:
            out.append(res)
            per_source[res.document.source] = n + 1
        if len(out) >= top_k:
            break
    return out


def retrieve_candidate_context(
    candidate_id: str,
    query: str,
    top_k: int = 8,
    source: Optional[str] = None,
    chunk_type: Optional[str] = None,
    project_name: Optional[str] = None,
    topic: Optional[str] = None,
    *,
    store: Optional[VectorStore] = None,
    provider: Optional[EmbeddingProvider] = None,
    config: Optional[RagConfig] = None,
    mode: str = "hybrid",
    diversity: bool = True,
    max_per_source: int = 3,
) -> List[SearchResult]:
    """Retrieve this candidate's most relevant chunks for a question.

    Args:
        candidate_id: owner of the knowledge (required; never optional).
        query: natural-language question; also the keyword source.
        top_k: max results returned.
        source/chunk_type/project_name/topic: metadata pre-filters applied
            inside both retrieval paths (lets the planner narrow retrieval).
        store/provider/config: injectable (tests pass fakes/memory stores).
        mode: "hybrid" (vector + keyword, RRF) | "vector" | "keyword".
        diversity/max_per_source: cap chunks kept per source.

    Returns:
        Ranked SearchResults (origin set, score lower-is-better), or []
        when the candidate has nothing matching. Provider/store outages
        raise (explicit failure) instead of returning fabricated content.
    """
    if not candidate_id or not str(candidate_id).strip():
        raise ValueError("candidate_id is required (isolation).")
    if not query or not str(query).strip():
        return []
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    mode = (mode or "hybrid").strip().lower()
    if mode not in ("hybrid", "vector", "keyword"):
        raise ValueError(f"Unknown retrieval mode {mode!r}. Expected 'hybrid', 'vector' or 'keyword'.")

    config = config or RagConfig.from_env()
    provider = provider or get_embedding_provider()
    store_mode = os.getenv("RAG_STORE_MODE", "auto").strip().lower()
    store = store or get_vector_store(mode=store_mode, provider=provider, config=config)
    filters = SearchFilters(source=source, chunk_type=chunk_type,
                            project_name=project_name, topic=topic)
    depth = max(top_k * OVERSAMPLE_FACTOR, OVERSAMPLE_MINIMUM)

    vector_hits: List[SearchResult] = []
    keyword_hits: List[SearchResult] = []
    if mode in ("hybrid", "vector"):
        query_vector = provider.embed_text(str(query))
        vector_hits = store.search(candidate_id, query_vector, k=depth, filters=filters)
    if mode in ("hybrid", "keyword"):
        terms = extract_terms(str(query))
        if terms:
            keyword_hits = store.keyword_search(candidate_id, terms, k=depth, filters=filters)

    if mode == "vector":
        ranked = vector_hits
    elif mode == "keyword":
        ranked = keyword_hits
    else:
        ranked = []
        for res, fused in rrf_fuse([vector_hits, keyword_hits]):
            res.score = 1.0 / (1.0 + fused)  # distance convention: lower = better
            res.origin = "hybrid"
            ranked.append(res)

    if diversity:
        ranked = apply_diversity(ranked, top_k, max_per_source)
    else:
        ranked = ranked[:top_k]
    return check_candidates_match(ranked, candidate_id)


def format_context(results: List[SearchResult], max_chars: int = 4000) -> str:
    """Render results as a cited evidence block for interview prompts.

    Empty input yields an explicit no-evidence marker (never fabricated text),
    so the interviewer falls back to general CS questions.
    """
    if not results:
        return "[No candidate-specific evidence retrieved.]"
    parts = []
    used = 0
    for i, r in enumerate(results, start=1):
        cite = r.document.project_name or r.document.topic or r.document.source
        chunk = f"[{i}] ({r.document.source}/{r.document.chunk_type}, {cite}): {r.document.text}"
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n".join(parts) if parts else "[No candidate-specific evidence retrieved.]"


__all__ = [
    "RRF_K",
    "STOPWORDS",
    "retrieve_candidate_context",
    "extract_terms",
    "rrf_fuse",
    "apply_diversity",
    "format_context",
]
