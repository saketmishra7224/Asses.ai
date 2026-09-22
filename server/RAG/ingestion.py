"""Candidate knowledge ingestion (Phase 2).

Turns the already-enriched upload profile into semantic RagDocuments:

  resume text   -> section-aware chunks (education/skills/experience/...)
  GitHub repos  -> overview + README chunks + technology statement per repo
  LeetCode stats-> one natural-language statement per solved topic + summary

Nothing here touches the network: it consumes the profile dict produced by
``main._enrich_profile`` (which reuses ``Scrapper/scrap.py`` fetchers).
Embedding + vector upsert happen in :func:`ingest_candidate_profile`.

Document IDs are deterministic (source + stable key), so re-ingesting the
same profile upserts (updates) instead of duplicating. Pass ``replace=True``
(e.g. fresh re-upload) to first purge the candidate's old chunks.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import EmbeddingProvider, get_embedding_provider
from .index import DimensionMismatchError, RagConfig, RagUnavailableError, VectorStore, get_vector_store
from .models import RagDocument

CHUNK_CHARS = int(os.getenv("RAG_CHUNK_CHARS", "1200"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
EMBED_BATCH = 32

LEETCODE_BUCKETS = ("fundamental", "intermediate", "advanced")

# heading keyword -> section category (first match wins; checked in order)
SECTION_KEYWORDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("education", ("education", "academic", "qualification", "university", "college", "degree")),
    ("experience", ("experience", "employment", "work history", "internship", "professional background")),
    ("skills", ("skill", "technical skills", "technologies", "tech stack", "competenc", "tools", "languages")),
    ("projects", ("project", "portfolio", "personal work")),
    ("achievements", ("achievement", "accomplishment", "award", "honor", "recognition")),
    ("certifications", ("certification", "certificate", "licen", "course", "training")),
    ("summary", ("summary", "objective", "profile", "about me", "overview")),
]


def _is_heading(line: str) -> Optional[str]:
    """Return a section category if the line looks like a resume heading."""
    s = line.strip()
    if not s or len(s) > 60 or s[-1:] in ".:;,":
        return None
    if s.startswith("#"):  # markdown heading
        s = s.lstrip("#").strip()
        if not s:
            return None
    low = s.lower().rstrip(":")
    for section, keywords in SECTION_KEYWORDS:
        for kw in keywords:
            if low == kw or low.startswith(kw + " ") or low.startswith(kw + ":"):
                return section
    if s.isupper() and len(s.split()) <= 5:  # "WORK EXPERIENCE", "SKILLS"
        for section, keywords in SECTION_KEYWORDS:
            if any(kw.split()[0] in low for kw in keywords):
                return section
    return None


def split_resume_sections(text: str) -> List[Tuple[str, str]]:
    """Split resume text into (section, body) blocks on detected headings.

    Text before the first heading is labeled 'summary'. Text with no
    headings at all becomes a single ('other', text) block.
    """
    if not text or not text.strip():
        return []
    sections: List[Tuple[str, List[str]]] = []
    current: Optional[List] = None
    for raw_line in text.splitlines():
        section = _is_heading(raw_line)
        if section is not None:
            current = [section, []]
            sections.append(current)
        else:
            if current is None:
                current = ["summary", []]
                sections.append(current)
            if raw_line.strip():
                current[1].append(raw_line.strip())
    out = [(name, "\n".join(lines)) for name, lines in sections if lines]
    if not out:
        return [("other", text.strip())]
    return out


def window_text(text: str, size: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sliding character windows on whitespace boundaries (overlap keeps context)."""
    text = " ".join(text.split())
    if len(text) <= size:
        return [text] if text else []
    windows, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut
        windows.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [w for w in windows if w]


def chunk_resume_text(text: str, candidate_id: str) -> List[RagDocument]:
    """Semantic resume chunks: one document per section window.

    IDs (``resume|<section>|<n>`` with a per-section counter) are
    deterministic: re-ingesting an unchanged resume upserts the same keys
    instead of duplicating — even when a section name repeats.
    """
    docs = []
    seen: Dict[str, int] = {}
    for section, body in split_resume_sections(text):
        for window in window_text(body, CHUNK_CHARS, CHUNK_OVERLAP):
            n = seen.get(section, 0)
            seen[section] = n + 1
            docs.append(RagDocument(
                candidate_id=candidate_id,
                document_id=f"resume|{section}|{n}",
                source="resume",
                chunk_type="section",
                section=section,
                text=window,
            ))
    return docs


def parse_repo_slug(url: str) -> str:
    """'https://github.com/owner/repo' -> 'owner/repo', else ''."""
    try:
        parts = (url or "").strip().strip("/").split("/")
        if len(parts) >= 5 and parts[2].lower().endswith("github.com"):
            return f"{parts[3]}/{parts[4]}"
    except Exception:
        pass
    return ""


def split_markdown_sections(readme: str) -> List[Tuple[str, str]]:
    """Split a README on markdown headings; returns (heading, body)."""
    if not readme or not readme.strip():
        return []
    sections: List[Tuple[str, List[str]]] = []
    current: Optional[List] = None
    for raw_line in readme.splitlines():
        s = raw_line.strip()
        if s.startswith("#"):
            heading = s.lstrip("#").strip().lower()[:60] or "section"
            current = [heading, []]
            sections.append(current)
        else:
            if current is None:
                current = ["overview", []]
                sections.append(current)
            if s:
                current[1].append(s)
    return [(h, "\n".join(b)) for h, b in sections if b] or [("overview", readme.strip())]


def chunk_github_repo(
    url: str,
    description: str,
    readme: str,
    language: str,
    candidate_id: str,
) -> List[RagDocument]:
    """Project-level chunks for one repo (bounded: no code indexing).

    Emits an overview doc, README section windows, and a technology
    statement when the primary language is known. Empty repos yield [].
    """
    slug = parse_repo_slug(url)
    name = slug.split("/")[-1] if slug else (url or "repo")
    description, readme, language = (description or "").strip(), (readme or "").strip(), (language or "").strip()
    if readme.startswith("Error:"):
        readme = ""
    if not description and not readme:
        return []  # unreachable/private repo: nothing worth embedding
    docs = [RagDocument(
        candidate_id=candidate_id,
        document_id=f"github|{slug or name}|overview",
        source="github",
        chunk_type="project",
        project_name=name,
        repository=slug,
        language=language,
        section="overview",
        text=f"{name}: {description or 'No description.'}"
             + (f" Primary language: {language}." if language else "")
             + (f" Repository: {url}." if url else ""),
    )]
    for i, (heading, body) in enumerate(split_markdown_sections(readme)):
        for j, window in enumerate(window_text(f"{name} - {heading}: {body}")):
            docs.append(RagDocument(
                candidate_id=candidate_id,
                document_id=f"github|{slug or name}|readme|{i}-{j}",
                source="github",
                chunk_type="readme",
                project_name=name,
                repository=slug,
                language=language,
                section=heading,
                text=window,
            ))
    if language:
        docs.append(RagDocument(
            candidate_id=candidate_id,
            document_id=f"github|{slug or name}|technology",
            source="github",
            chunk_type="technology",
            project_name=name,
            repository=slug,
            language=language,
            section="technology",
            skill=language.lower(),
            text=f"The {name} project is primarily written in {language}.",
        ))
    return docs


def normalize_leetcode_stats(raw: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Accept the profile's leetcode_stats shapes and return {bucket: [tags]}.

    Handles {"tagProblemCounts": {...}}, {"matchedUser": {"tagProblemCounts": ...}},
    and raw JSON strings. Returns {} when nothing usable is present.
    """
    if isinstance(raw, str):
        try:
            import json

            raw = json.loads(raw) if raw.strip().startswith("{") else None
        except Exception:
            return {}
    if not isinstance(raw, dict):
        return {}
    counts = raw.get("tagProblemCounts") or raw.get("matchedUser", {}).get("tagProblemCounts") or {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for bucket in LEETCODE_BUCKETS:
        tags = counts.get(bucket) or []
        clean = [
            {"tagName": str(t.get("tagName", "")), "tagSlug": str(t.get("tagSlug", "")),
             "problemsSolved": int(t.get("problemsSolved", 0) or 0)}
            for t in tags if isinstance(t, dict) and t.get("tagSlug")
        ]
        if clean:
            out[bucket] = clean
    return out


def chunk_leetcode_stats(stats: Any, candidate_id: str) -> List[RagDocument]:
    """One natural-language statement per solved topic + a totals summary.

    Stored metadata (topic/category/problems_solved) supports filtered
    retrieval like 'candidate's graph strengths'; the original structured
    stats stay untouched in the session profile for the UI.
    """
    buckets = normalize_leetcode_stats(stats)
    if not buckets:
        return []
    docs: List[RagDocument] = []
    total = 0
    best: List[tuple] = []
    for bucket in LEETCODE_BUCKETS:
        for tag in buckets.get(bucket, []):
            n = tag["problemsSolved"]
            if n <= 0:
                continue
            total += n
            best.append((n, tag["tagName"]))
            docs.append(RagDocument(
                candidate_id=candidate_id,
                document_id=f"leetcode|{bucket}|{tag['tagSlug']}",
                source="leetcode",
                chunk_type="topic",
                topic=tag["tagSlug"],
                skill=tag["tagSlug"],
                category=bucket,
                problems_solved=n,
                text=f"Candidate has solved {n} {tag['tagName']} problems on LeetCode ({bucket} category).",
            ))
    if total:
        best.sort(reverse=True)
        top = ", ".join(f"{name} ({n})" for n, name in best[:3])
        docs.append(RagDocument(
            candidate_id=candidate_id,
            document_id="leetcode|summary",
            source="leetcode",
            chunk_type="profile",
            problems_solved=total,
            text=f"Candidate has solved {total} LeetCode problems in total. "
                 f"Strongest areas: {top}.",
        ))
    return docs


def build_candidate_documents(profile: Dict[str, Any], candidate_id: str) -> Tuple[List[RagDocument], List[str]]:
    """Pure chunking step: profile -> (documents, warnings). No network, no embeddings."""
    if not candidate_id:
        raise ValueError("candidate_id is required.")
    profile = profile or {}
    docs: List[RagDocument] = []
    warnings: List[str] = []

    resume_text = profile.get("resume_text") or ""
    if resume_text.strip():
        try:
            docs.extend(chunk_resume_text(resume_text, candidate_id))
        except Exception as e:
            warnings.append(f"Resume chunking failed: {e}")
    else:
        warnings.append("No resume text available; resume chunks skipped.")

    repos = profile.get("github") or []
    if repos:
        try:
            for repo in repos:
                if not isinstance(repo, dict):
                    continue
                details = repo.get("details") or []
                # Supports both enriched entries {url, description, readme, language}
                # and legacy UI items {description, readme}.
                docs.extend(chunk_github_repo(
                    url=repo.get("url") or "",
                    description=repo.get("description") or (details[0] if len(details) > 0 else ""),
                    readme=repo.get("readme") or (details[1] if len(details) > 1 else ""),
                    language=repo.get("language") or (details[2] if len(details) > 2 else ""),
                    candidate_id=candidate_id,
                ))
        except Exception as e:
            warnings.append(f"GitHub chunking failed: {e}")
    else:
        warnings.append("No GitHub data available; GitHub chunks skipped.")

    leetcode_stats = profile.get("leetcode_stats")
    if leetcode_stats is None and profile.get("leetcode_raw"):
        leetcode_stats = profile.get("leetcode_raw")
    leet_docs: List[RagDocument] = []
    try:
        leet_docs = chunk_leetcode_stats(leetcode_stats, candidate_id)
    except Exception as e:
        warnings.append(f"LeetCode chunking failed: {e}")
    if leet_docs:
        docs.extend(leet_docs)
    else:
        warnings.append("No usable LeetCode stats; LeetCode chunks skipped.")

    return docs, warnings


def _embed_all(provider: EmbeddingProvider, docs: List[RagDocument]) -> Tuple[List[Tuple[RagDocument, List[float]]], List[str]]:
    """Batch-embed with per-document fallback: one bad chunk never kills the batch."""
    warnings: List[str] = []
    embedded: List[Tuple[RagDocument, List[float]]] = []
    for i in range(0, len(docs), EMBED_BATCH):
        batch = docs[i:i + EMBED_BATCH]
        try:
            vectors = provider.embed_texts([d.text for d in batch])
            embedded.extend(zip(batch, vectors))
        except Exception as e:
            warnings.append(f"Batch embedding failed ({len(batch)} docs): {e}; retrying per document.")
            for doc in batch:
                try:
                    embedded.append((doc, provider.embed_text(doc.text)))
                except Exception as e2:
                    warnings.append(f"Skipping document {doc.document_id}: {e2}")
    return embedded, warnings


def ingest_candidate_profile(
    candidate_id: str,
    profile: Optional[Dict[str, Any]],
    *,
    store: Optional[VectorStore] = None,
    provider: Optional[EmbeddingProvider] = None,
    config: Optional[RagConfig] = None,
    replace: bool = False,
) -> Dict[str, Any]:
    """Full pipeline: normalize -> chunk -> embed -> upsert. Never raises
    (except ValueError for a missing candidate_id); failures are reported in
    the returned stats dict so upload/interview flows keep working.

    {
      "candidate_id": ...,
      "status": "ready" | "partial" | "failed" | "empty",
      "documents_created": int, "documents_updated": int,
      "sources": {"resume": int, "github": int, "leetcode": int},
      "warnings": [...],
    }
    """
    from .index import DimensionMismatchError, RagUnavailableError, get_vector_store

    result: Dict[str, Any] = {
        "candidate_id": candidate_id,
        "status": "failed",
        "documents_created": 0,
        "documents_updated": 0,
        "sources": {"resume": 0, "github": 0, "leetcode": 0},
        "warnings": [],
    }
    if not candidate_id or not str(candidate_id).strip():
        raise ValueError("candidate_id is required.")
    if not profile:
        result["warnings"].append("Empty profile; nothing to ingest.")
        result["status"] = "empty"
        return result

    try:
        provider = provider or get_embedding_provider()
    except Exception as e:
        result["warnings"].append(f"Embedding provider unavailable: {e}")
        return result
    try:
        if store is None:
            store_mode = os.getenv("RAG_STORE_MODE", "redis").strip().lower()
            store = get_vector_store(mode=store_mode, dimension=provider.dimension, config=config)
        status = store.ensure_index(provider.dimension)
        if not status.get("ok"):
            result["warnings"].append(f"Vector index not ready: {status.get('reason', status)}")
            return result
    except Exception as e:
        result["warnings"].append(f"Vector store unavailable: {e}")
        return result

    try:
        docs, warnings = build_candidate_documents(profile, candidate_id)
        result["warnings"].extend(warnings)
    except Exception as e:
        result["warnings"].append(f"Document building failed: {e}")
        return result
    if not docs:
        result["status"] = "empty"
        return result

    if replace:
        try:
            store.delete_candidate(candidate_id)
        except Exception as e:
            result["warnings"].append(f"Could not purge old chunks: {e}")

    embedded, embed_warnings = _embed_all(provider, docs)
    result["warnings"].extend(embed_warnings)
    if not embedded:
        return result

    # created vs updated: probe pre-existing keys before upserting
    # (skipped after a replace-purge, where everything is new).
    pre_existing = set()
    if not replace:
        prefix = store.config.key_prefix
        for doc, _ in embedded:
            try:
                if store.exists(doc.key(prefix)):
                    pre_existing.add(doc.key(prefix))
            except Exception:
                pass  # probe failure -> counted as created below
    try:
        keys = store.upsert_many(embedded)
    except (DimensionMismatchError, ValueError):
        raise
    except Exception as e:
        result["warnings"].append(f"Upsert failed: {e}")
        return result

    for (_, _), key in zip(embedded, keys):
        if key in pre_existing:
            result["documents_updated"] += 1
        else:
            result["documents_created"] += 1
    result["sources"] = _count_sources([d for d, _ in embedded])
    result["status"] = "ready" if not result["warnings"] else "partial"
    return result


def _count_sources(docs: List[RagDocument]) -> Dict[str, int]:
    counts = {"resume": 0, "github": 0, "leetcode": 0}
    for d in docs:
        if d.source in counts:
            counts[d.source] += 1
    return counts
