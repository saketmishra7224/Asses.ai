"""Core data models for the candidate RAG layer.

These are pure data structures with no network access: safe to import
anywhere, including unit tests without API keys or Redis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Where a chunk of candidate knowledge came from.
SOURCES = ("resume", "github", "leetcode", "summary")

# Granularity / role of a chunk:
#   section    - a resume section (education/skills/experience/...) or doc section
#   project    - GitHub repo overview
#   readme     - a chunk of a README file
#   technology - repo primary language / stack statement
#   overview   - generic overview chunk
#   topic      - one LeetCode topic (tag) statement
#   profile    - aggregate profile summary (e.g. LeetCode totals)
#   summary    - aggregate candidate summary
CHUNK_TYPES = ("section", "project", "readme", "technology", "overview", "topic", "profile", "summary")


@dataclass
class RagDocument:
    """One retrievable unit of candidate knowledge.

    Identity is (candidate_id, document_id); storage key is
    ``{prefix}{candidate_id}:{document_id}`` (default prefix ``rag:``).
    """

    candidate_id: str
    document_id: str
    source: str
    text: str
    chunk_type: str = "section"
    project_name: str = ""
    repository: str = ""
    topic: str = ""
    language: str = ""
    section: str = ""
    skill: str = ""
    category: str = ""
    problems_solved: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def validate(self) -> "RagDocument":
        if not self.candidate_id or not str(self.candidate_id).strip():
            raise ValueError("RagDocument requires a non-empty candidate_id.")
        if not self.document_id or not str(self.document_id).strip():
            raise ValueError("RagDocument requires a non-empty document_id.")
        if not self.text or not str(self.text).strip():
            raise ValueError("RagDocument requires non-empty text.")
        if self.source not in SOURCES:
            raise ValueError(f"Unknown source {self.source!r}. Expected one of {SOURCES}.")
        if self.chunk_type not in CHUNK_TYPES:
            raise ValueError(f"Unknown chunk_type {self.chunk_type!r}. Expected one of {CHUNK_TYPES}.")
        return self

    def key(self, prefix: str = "rag:") -> str:
        return f"{prefix}{self.candidate_id}:{self.document_id}"

    def to_mapping(self) -> Dict[str, str]:
        """String field mapping for Redis HASH storage (embedding set separately)."""
        self.validate()
        return {
            "candidate_id": str(self.candidate_id),
            "document_id": str(self.document_id),
            "source": self.source,
            "chunk_type": self.chunk_type,
            "project_name": self.project_name or "",
            "repository": self.repository or "",
            "topic": self.topic or "",
            "language": self.language or "",
            "section": self.section or "",
            "skill": self.skill or "",
            "category": self.category or "",
            "problems_solved": str(int(self.problems_solved or 0)),
            "text": self.text,
            "created_at": str(self.created_at),
            "updated_at": str(self.updated_at),
        }

    @classmethod
    def from_mapping(cls, mapping: Dict[Any, Any]) -> "RagDocument":
        """Rebuild from a Redis HASH mapping (str or bytes values; ignores embedding blob)."""
        def s(v: Any) -> str:
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return str(v)

        def f(v: Any) -> float:
            try:
                return float(s(v))
            except (TypeError, ValueError):
                return 0.0

        def n(v: Any) -> int:
            try:
                return int(float(s(v)))
            except (TypeError, ValueError):
                return 0

        return cls(
            candidate_id=s(mapping.get("candidate_id", "")),
            document_id=s(mapping.get("document_id", "")),
            source=s(mapping.get("source", "")) or "resume",
            text=s(mapping.get("text", "")),
            chunk_type=s(mapping.get("chunk_type", "")) or "section",
            project_name=s(mapping.get("project_name", "")),
            repository=s(mapping.get("repository", "")),
            topic=s(mapping.get("topic", "")),
            language=s(mapping.get("language", "")),
            section=s(mapping.get("section", "")),
            skill=s(mapping.get("skill", "")),
            category=s(mapping.get("category", "")),
            problems_solved=n(mapping.get("problems_solved", 0)),
            created_at=f(mapping.get("created_at", 0.0)),
            updated_at=f(mapping.get("updated_at", 0.0)),
        )


@dataclass
class SearchFilters:
    """Optional metadata pre-filters for retrieval. All are exact matches
    except `text`, which is a full-text / substring match on chunk text."""

    source: Optional[str] = None
    chunk_type: Optional[str] = None
    project_name: Optional[str] = None
    repository: Optional[str] = None
    topic: Optional[str] = None
    language: Optional[str] = None
    section: Optional[str] = None
    skill: Optional[str] = None
    category: Optional[str] = None
    text: Optional[str] = None

    def active(self) -> Dict[str, str]:
        return {k: v for k, v in {
            "source": self.source,
            "chunk_type": self.chunk_type,
            "project_name": self.project_name,
            "repository": self.repository,
            "topic": self.topic,
            "language": self.language,
            "section": self.section,
            "skill": self.skill,
            "category": self.category,
            "text": self.text,
        }.items() if v}


@dataclass
class SearchResult:
    """A retrieved chunk with its relevance score.

    ``score`` is a *distance*: lower means more relevant, for every origin
    (cosine distance for "vector", inverted match strength for "keyword",
    inverted RRF for "hybrid"). ``origin`` records which path produced the
    result: "vector" | "keyword" | "hybrid".
    """

    document: RagDocument
    score: float
    key: str = ""
    origin: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "score": self.score,
            "origin": self.origin,
            "candidate_id": self.document.candidate_id,
            "document_id": self.document.document_id,
            "source": self.document.source,
            "chunk_type": self.document.chunk_type,
            "project_name": self.document.project_name,
            "repository": self.document.repository,
            "topic": self.document.topic,
            "language": self.document.language,
            "text": self.document.text,
        }


def check_candidates_match(results: List[SearchResult], candidate_id: str) -> List[SearchResult]:
    """Isolation guard: return only results belonging to `candidate_id`.

    Defense-in-depth for the mandatory server-side session filter — callers
    (and tests) use this to prove no cross-candidate leakage.
    """
    return [r for r in results if r.document.candidate_id == candidate_id]
