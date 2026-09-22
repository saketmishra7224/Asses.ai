"""Redis Stack vector storage for candidate knowledge.

Design notes (full rationale in ``server/RAG/README.md``):

* Storage is Redis HASH (binary-safe ``FLOAT32`` embedding blob + plain
  string metadata), indexed by one global RediSearch index.
* Candidate isolation has two layers: every chunk carries a ``candidate_id``
  TAG that the query path MUST filter on, and keys are namespaced
  ``{prefix}{candidate_id}:{document_id}``.
* :class:`RedisVectorStore` is the production backend;
  :class:`InMemoryVectorStore` implements identical filtering semantics for
  unit tests and works without Redis or API keys.
* Importing this module performs no network I/O. All connection problems
  surface as status dicts (:func:`ensure_index`) or
  :class:`RagUnavailableError`, never as import-time crashes.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import (
    RagDocument,
    SearchFilters,
    SearchResult,
    check_candidates_match,
)


class RagUnavailableError(Exception):
    """Raised when vector storage is unreachable or unusable at call time."""


class DimensionMismatchError(RagUnavailableError):
    """Raised when data dimensionality conflicts with the existing index."""


# ------------------------------------------------------------------ config --
@dataclass
class RagConfig:
    """All tunables; env-backed with explicit overrides (tests use their own)."""

    index_name: str = "idx:candidate_chunks"
    key_prefix: str = "rag:"
    top_k: int = 4
    ttl_seconds: int = 604800  # 7 days; 0/None disables expiry
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_runtime: int = 10
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_socket_timeout: float = 2.0

    @classmethod
    def from_env(cls, **overrides: Any) -> "RagConfig":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)))
            except ValueError:
                return default

        cfg = cls(
            index_name=os.getenv("RAG_INDEX_NAME", "idx:candidate_chunks"),
            key_prefix=os.getenv("RAG_KEY_PREFIX", "rag:"),
            top_k=_int("RAG_TOP_K", 4),
            ttl_seconds=_int("RAG_TTL_SECONDS", 604800),
            hnsw_m=_int("RAG_HNSW_M", 16),
            hnsw_ef_construction=_int("RAG_HNSW_EF_CONSTRUCTION", 200),
            hnsw_ef_runtime=_int("RAG_HNSW_EF_RUNTIME", 10),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=_int("REDIS_PORT", 6379),
        )
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ValueError(f"Unknown RagConfig field: {k}")
            setattr(cfg, k, v)
        return cfg

    def meta_key(self) -> str:
        return f"{self.key_prefix}_meta:{self.index_name}"


# ----------------------------------------------------------------- helpers --
def vector_to_blob(vector: List[float], dimension: int) -> bytes:
    """Serialize to the FLOAT32 blob the RediSearch VECTOR field expects."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != dimension:
        raise DimensionMismatchError(
            f"Vector dim {arr.shape} does not match index dim {dimension}."
        )
    return arr.tobytes()


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 1.0
    return float(1.0 - float(np.dot(a, b)) / denom)


_TAG_ESCAPE = set(',.<>{}[]"\\:;!@#$%^&*()-+=~|/? ')


def escape_tag(value: str) -> str:
    """Escape a value for use inside a RediSearch TAG query {...}."""
    return "".join(f"\\{ch}" if ch in _TAG_ESCAPE else ch for ch in str(value))


def sanitize_text_term(term: str) -> str:
    """Reduce a keyword to characters safe inside a RediSearch TEXT query.

    Keeps letters, digits and + # . - _ (so C++, Next.js, node.js survive);
    drops anything shorter than 2 chars afterwards.
    """
    cleaned = re.sub(r"[^A-Za-z0-9_+#.\-]", "", str(term or ""))
    return cleaned if len(cleaned) >= 2 else ""


def connect_redis(config: Optional[RagConfig] = None):
    """Open a bytes-mode Redis client, or return None when unreachable.

    Never raises: connection problems (no server, no network, refused)
    return None so callers degrade gracefully. A raw-socket pre-probe with a
    short timeout keeps this fast (~0.5 s) when Redis isn't running, avoiding
    redis-py's longer internal retry/backoff on some platforms.
    """
    config = config or RagConfig.from_env()
    import socket as _socket

    try:
        probe = _socket.create_connection((config.redis_host, config.redis_port), timeout=0.5)
        probe.close()
    except Exception:
        return None
    try:
        from redis import Redis

        client = Redis(
            host=config.redis_host,
            port=config.redis_port,
            decode_responses=False,  # binary-safe: embedding blobs are raw bytes
            socket_connect_timeout=min(config.redis_socket_timeout, 2.0),
            socket_timeout=config.redis_socket_timeout,
            health_check_interval=0,
        )
        client.ping()
        return client
    except Exception:
        return None


def redis_has_search(client) -> bool:
    """True when the server offers the RediSearch module (Redis Stack)."""
    try:
        modules = client.module_list() or []
        names = set()
        for m in modules:
            name = m.get(b"name", b"") if isinstance(m, dict) else b""
            names.add(name.decode("utf-8", errors="ignore").lower())
        return "search" in names
    except Exception:
        return False


# Cache of (host, port) -> (has_search, timestamp). A MODULE LIST round-trip
# per interview turn would add latency and log spam when Redis lacks the
# module, so the check is cached briefly (modules don't change at runtime
# in practice; worst case the cache expires and re-checks).
_SEARCH_SUPPORT_CACHE: Dict[Tuple[str, int], Tuple[bool, float]] = {}
_SEARCH_SUPPORT_TTL = 120.0
_WARNED_NO_SEARCH: set = set()


def has_vector_search(client, host: str = "localhost", port: int = 6379) -> bool:
    """Cached wrapper around redis_has_search. Never raises (False on error)."""
    key = (host, port)
    now = time.time()
    hit = _SEARCH_SUPPORT_CACHE.get(key)
    if hit is not None and now - hit[1] < _SEARCH_SUPPORT_TTL:
        return hit[0]
    ok = redis_has_search(client)
    _SEARCH_SUPPORT_CACHE[key] = (ok, now)
    return ok


def _warn_no_search_once(host: str, port: int) -> None:
    """Log the plain-Redis guidance exactly once per process."""
    if (host, port) not in _WARNED_NO_SEARCH:
        _WARNED_NO_SEARCH.add((host, port))
        print(f"[rag] Redis at {host}:{port} has no RediSearch module (plain Redis?). "
              "Vector search needs the redis/redis-stack image — see docker-compose.yml. "
              "Interviews continue with general questions until then.")


# ------------------------------------------------------------------ ABC ----
class VectorStore(ABC):
    """Storage backend contract. Filtering semantics must match across backends."""

    backend = "base"

    @abstractmethod
    def ensure_index(self, dimension: int) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, doc: RagDocument, embedding: List[float]) -> str:
        raise NotImplementedError

    @abstractmethod
    def upsert_many(self, items: List[Tuple[RagDocument, List[float]]]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        candidate_id: str,
        query_vector: List[float],
        k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """KNN search restricted to `candidate_id`. Never returns other
        candidates' chunks (enforced + re-checked via check_candidates_match)."""
        raise NotImplementedError

    @abstractmethod
    def keyword_search(
        self,
        candidate_id: str,
        terms: List[str],
        k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Lexical search restricted to `candidate_id`: chunks whose text
        matches any of `terms`. Same isolation guarantees as search()."""
        raise NotImplementedError

    @abstractmethod
    def delete_candidate(self, candidate_id: str) -> int:
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        """True when a chunk key already exists (used for created/updated stats)."""
        raise NotImplementedError

    @abstractmethod
    def count(self, candidate_id: Optional[str] = None) -> int:
        raise NotImplementedError


# ----------------------------------------------------------------- memory --
_SHARED_IN_MEMORY_STORES: Dict[Tuple[int, str], InMemoryVectorStore] = {}


def get_shared_in_memory_store(
    dimension: int,
    config: Optional[RagConfig] = None,
) -> InMemoryVectorStore:
    """Return a process-level shared InMemoryVectorStore for a given dimension and prefix.

    Ensures that chunks stored during profile ingestion remain available across requests
    during the server lifetime without needing Docker or Redis Stack.
    """
    config = config or RagConfig.from_env()
    key = (dimension, config.key_prefix)
    if key not in _SHARED_IN_MEMORY_STORES:
        _SHARED_IN_MEMORY_STORES[key] = InMemoryVectorStore(dimension, config)
    return _SHARED_IN_MEMORY_STORES[key]


class InMemoryVectorStore(VectorStore):
    """Brute-force backend with identical filtering semantics. For unit tests
    and offline development — no Redis, no keys, no network."""

    backend = "memory"

    def __init__(self, dimension: int, config: Optional[RagConfig] = None) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.dimension = dimension
        self.config = config or RagConfig.from_env()
        self._docs: Dict[str, RagDocument] = {}
        self._vecs: Dict[str, np.ndarray] = {}

    # -- lifecycle ---------------------------------------------------
    def ensure_index(self, dimension: int) -> Dict[str, Any]:
        if dimension != self.dimension:
            return {
                "ok": False,
                "status": "dimension_mismatch",
                "expected": self.dimension,
                "found": dimension,
                "reason": "Re-create the store with the matching dimension.",
            }
        return {"ok": True, "status": "ready", "backend": self.backend}

    # -- writes ------------------------------------------------------
    def upsert(self, doc: RagDocument, embedding: List[float]) -> str:
        doc.validate()
        arr = np.asarray(embedding, dtype=np.float32)
        if arr.ndim != 1 or arr.shape[0] != self.dimension:
            raise DimensionMismatchError(
                f"Vector dim {arr.shape} does not match store dim {self.dimension}."
            )
        key = doc.key(self.config.key_prefix)
        self._docs[key] = doc
        self._vecs[key] = arr
        return key

    def upsert_many(self, items: List[Tuple[RagDocument, List[float]]]) -> List[str]:
        return [self.upsert(doc, vec) for doc, vec in items]

    def delete_candidate(self, candidate_id: str) -> int:
        if not candidate_id:
            raise ValueError("candidate_id is required.")
        prefix = f"{self.config.key_prefix}{candidate_id}:"
        doomed = [k for k in self._docs if k.startswith(prefix)]
        for k in doomed:
            del self._docs[k]
            del self._vecs[k]
        return len(doomed)

    def exists(self, key: str) -> bool:
        return key in self._docs

    def count(self, candidate_id: Optional[str] = None) -> int:
        if candidate_id is None:
            return len(self._docs)
        prefix = f"{self.config.key_prefix}{candidate_id}:"
        return sum(1 for k in self._docs if k.startswith(prefix))

    # -- reads -------------------------------------------------------
    def _matches(self, doc: RagDocument, filters: SearchFilters) -> bool:
        f = filters.active()
        for field in ("source", "chunk_type", "project_name", "repository", "topic",
                      "language", "section", "skill", "category"):
            if field in f and getattr(doc, field) != f[field]:
                return False
        if "text" in f and f["text"].lower() not in (doc.text or "").lower():
            return False
        return True

    def search(
        self,
        candidate_id: str,
        query_vector: List[float],
        k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        if not candidate_id:
            raise ValueError("candidate_id is required (isolation).")
        query = np.asarray(query_vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != self.dimension:
            raise DimensionMismatchError("Query vector dim does not match store dim.")
        k = k or self.config.top_k
        filters = filters or SearchFilters()
        prefix = f"{self.config.key_prefix}{candidate_id}:"
        scored = []
        for key, doc in self._docs.items():
            if not key.startswith(prefix):
                continue
            if not self._matches(doc, filters):
                continue
            scored.append(SearchResult(document=doc, score=cosine_distance(query, self._vecs[key]),
                                       key=key, origin="vector"))
        scored.sort(key=lambda r: r.score)
        return check_candidates_match(scored[:k], candidate_id)

    def keyword_search(
        self,
        candidate_id: str,
        terms: List[str],
        k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        if not candidate_id:
            raise ValueError("candidate_id is required (isolation).")
        terms = [t.strip().lower() for t in (terms or []) if t and t.strip()]
        if not terms:
            return []
        k = k or self.config.top_k
        filters = filters or SearchFilters()
        prefix = f"{self.config.key_prefix}{candidate_id}:"
        scored = []
        for key, doc in self._docs.items():
            if not key.startswith(prefix):
                continue
            if not self._matches(doc, filters):
                continue
            hay = (doc.text or "").lower()
            matched = sum(1 for t in terms if t in hay)
            if not matched:
                continue
            # distance-like score: lower = more terms matched
            scored.append(SearchResult(document=doc, score=1.0 - matched / len(terms),
                                       key=key, origin="keyword"))
        scored.sort(key=lambda r: r.score)
        return check_candidates_match(scored[:k], candidate_id)


# ------------------------------------------------------------------ redis --
_INDEX_META_VERSION = 2  # bump when the indexed schema changes (forces rebuild guidance)


def build_index_fields(dimension: int, config: Optional[RagConfig] = None):
    """RediSearch field list for the candidate index (offline-safe, unit-tested).

    HASH storage: metadata as plain strings, embedding as a FLOAT32 blob.
    """
    # redis-py >= 7 keeps field classes in redis.commands.search.field
    # (older: redis.commands.search.field_definition).
    try:
        from redis.commands.search.field import (
            NumericField,
            TagField,
            TextField,
            VectorField,
        )
    except ImportError:  # pragma: no cover - redis-py < 7 layout
        from redis.commands.search.field_definition import (  # type: ignore
            NumericField,
            TagField,
            TextField,
            VectorField,
        )

    config = config or RagConfig.from_env()
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    return [
        TagField("candidate_id"),
        TagField("source"),
        TagField("chunk_type"),
        TagField("project_name"),
        TagField("repository"),
        TagField("topic"),
        TagField("language"),
        TagField("section"),
        TagField("skill"),
        TagField("category"),
        TagField("document_id"),
        TextField("text"),
        NumericField("created_at"),
        NumericField("updated_at"),
        NumericField("problems_solved"),
        VectorField(
            "embedding",
            "HNSW",
            {
                "TYPE": "FLOAT32",
                "DIM": dimension,
                "DISTANCE_METRIC": "COSINE",
                "M": config.hnsw_m,
                "EF_CONSTRUCTION": config.hnsw_ef_construction,
                "EF_RUNTIME": config.hnsw_ef_runtime,
            },
        ),
    ]


class RedisVectorStore(VectorStore):
    """Production backend on Redis Stack (RediSearch + HASH storage)."""

    backend = "redis"

    def __init__(self, client, config: Optional[RagConfig] = None, dimension: Optional[int] = None) -> None:
        if client is None:
            raise RagUnavailableError("No Redis client (server unreachable?).")
        self.client = client
        self.config = config or RagConfig.from_env()
        self.dimension = dimension  # bound on first ensure_index if None

    # -- index lifecycle ---------------------------------------------
    def _ft(self):
        return self.client.ft(self.config.index_name)

    def ensure_index(self, dimension: int) -> Dict[str, Any]:
        """Idempotent creation. Safe to call on every startup.

        Returns a status dict (never raises for connection/schema states):
        created | exists | unavailable | unsupported | dimension_mismatch.
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        try:
            from redis.commands.search.index_definition import IndexDefinition, IndexType
            from redis.exceptions import ResponseError
        except ImportError as e:  # redis-py too old for search support
            return {"ok": False, "status": "unsupported", "reason": f"redis-py lacks search API: {e}"}

        try:
            if not redis_has_search(self.client):
                return {
                    "ok": False,
                    "status": "unsupported",
                    "reason": "Redis server has no RediSearch module (need Redis Stack).",
                }
        except Exception as e:
            return {"ok": False, "status": "unavailable", "reason": str(e)}

        fields = build_index_fields(dimension, self.config)
        definition = IndexDefinition(prefix=[self.config.key_prefix], index_type=IndexType.HASH)
        try:
            self._ft().create_index(fields, definition=definition)
            self._write_meta(dimension)
            self.dimension = dimension
            return {"ok": True, "status": "created", "index": self.config.index_name, "dimension": dimension}
        except ResponseError as e:
            if "already exists" not in str(e).lower():
                return {"ok": False, "status": "error", "reason": str(e)}
            return self._verify_existing(dimension)
        except Exception as e:
            return {"ok": False, "status": "unavailable", "reason": f"{type(e).__name__}: {e}"}

    def _write_meta(self, dimension: int) -> None:
        try:
            self.client.set(
                self.config.meta_key(),
                json.dumps({"version": _INDEX_META_VERSION, "dimension": dimension,
                            "index": self.config.index_name, "ts": time.time()}),
            )
        except Exception:
            pass  # meta is advisory; the index itself is authoritative

    def _verify_existing(self, dimension: int) -> Dict[str, Any]:
        try:
            raw = self.client.get(self.config.meta_key())
            if raw:
                meta = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
                if int(meta.get("dimension", -1)) != dimension:
                    return {
                        "ok": False,
                        "status": "dimension_mismatch",
                        "expected": dimension,
                        "found": int(meta.get("dimension", -1)),
                        "reason": (
                            "Existing index was built with a different embedding dim. "
                            f"Drop it (FT.DROPINDEX {self.config.index_name} DD) and re-run "
                            "ensure_index, or set RAG_EMBEDDING_DIM to match."
                        ),
                    }
                if int(meta.get("version", 1)) != _INDEX_META_VERSION:
                    return {
                        "ok": False,
                        "status": "schema_mismatch",
                        "expected": _INDEX_META_VERSION,
                        "found": int(meta.get("version", 1)),
                        "reason": (
                            "Existing index uses an older chunk schema. "
                            f"Drop it (FT.DROPINDEX {self.config.index_name} DD) and re-run "
                            "ensure_index to rebuild with the current schema."
                        ),
                    }
        except Exception:
            pass
        self.dimension = dimension
        return {"ok": True, "status": "exists", "index": self.config.index_name, "dimension": dimension}

    # -- writes ------------------------------------------------------
    def _require_dim(self) -> int:
        if not self.dimension:
            raise RagUnavailableError("Index dimension unknown: call ensure_index(dimension) first.")
        return self.dimension

    def upsert(self, doc: RagDocument, embedding: List[float]) -> str:
        doc.validate()
        blob = vector_to_blob(embedding, self._require_dim())
        key = doc.key(self.config.key_prefix)
        mapping = doc.to_mapping()
        mapping["embedding"] = blob
        try:
            pipe = self.client.pipeline(transaction=False)
            pipe.hset(key, mapping=mapping)
            if self.config.ttl_seconds:
                pipe.expire(key, self.config.ttl_seconds)
            pipe.execute()
        except Exception as e:
            raise RagUnavailableError(f"upsert failed: {type(e).__name__}: {e}")
        return key

    def upsert_many(self, items: List[Tuple[RagDocument, List[float]]]) -> List[str]:
        dim = self._require_dim()
        try:
            pipe = self.client.pipeline(transaction=False)
            keys = []
            for doc, vec in items:
                doc.validate()
                mapping = doc.to_mapping()
                mapping["embedding"] = vector_to_blob(vec, dim)
                key = doc.key(self.config.key_prefix)
                pipe.hset(key, mapping=mapping)
                if self.config.ttl_seconds:
                    pipe.expire(key, self.config.ttl_seconds)
                keys.append(key)
            pipe.execute()
        except (DimensionMismatchError, ValueError):
            raise
        except Exception as e:
            raise RagUnavailableError(f"upsert_many failed: {type(e).__name__}: {e}")
        return keys

    def delete_candidate(self, candidate_id: str) -> int:
        if not candidate_id:
            raise ValueError("candidate_id is required.")
        pattern = f"{self.config.key_prefix}{candidate_id}:*"
        try:
            keys = list(self.client.scan_iter(match=pattern, count=500))
            if keys:
                self.client.delete(*keys)
            return len(keys)
        except Exception as e:
            raise RagUnavailableError(f"delete_candidate failed: {type(e).__name__}: {e}")

    def exists(self, key: str) -> bool:
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            raise RagUnavailableError(f"exists failed: {type(e).__name__}: {e}")

    def count(self, candidate_id: Optional[str] = None) -> int:
        from redis.commands.search.query import Query

        try:
            if candidate_id:
                q = Query(f"@candidate_id:{{{escape_tag(candidate_id)}}}").paging(0, 1).no_content()
            else:
                q = Query("*").paging(0, 1).no_content()
            return int(self._ft().search(q).total)
        except Exception as e:
            raise RagUnavailableError(f"count failed: {type(e).__name__}: {e}")

    # -- reads -------------------------------------------------------
    def _filter_expr(self, filters: SearchFilters) -> str:
        parts = []
        for field in ("source", "chunk_type", "project_name", "repository", "topic",
                      "language", "section", "skill", "category"):
            value = getattr(filters, field)
            if value:
                parts.append(f"@{field}:{{{escape_tag(value)}}}")
        if filters.text:
            safe = str(filters.text).replace('"', "").strip()
            if safe:
                parts.append(f'@text:"{safe}"')
        return " ".join(parts)

    def search(
        self,
        candidate_id: str,
        query_vector: List[float],
        k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        if not candidate_id:
            raise ValueError("candidate_id is required (isolation).")
        from redis.commands.search.query import Query
        from redis.exceptions import ResponseError

        dim = self._require_dim()
        blob = vector_to_blob(query_vector, dim)
        k = k or self.config.top_k
        filters = filters or SearchFilters()
        base = f"@candidate_id:{{{escape_tag(candidate_id)}}}"
        extra = self._filter_expr(filters)
        expr = f"({base} {extra})" if extra else f"({base})"
        query = (
            Query(f"{expr}=>[KNN {k} @embedding $vec AS vector_score]")
            .sort_by("vector_score")
            .paging(0, k)
            .dialect(2)
        )
        try:
            res = self._ft().search(query, query_params={"vec": blob})
        except ResponseError as e:
            if "no such index" in str(e).lower():
                raise RagUnavailableError(
                    f"Index {self.config.index_name} does not exist: call ensure_index() first."
                )
            raise RagUnavailableError(f"search failed: {e}")
        except Exception as e:
            raise RagUnavailableError(f"search failed: {type(e).__name__}: {e}")

        out = []
        for d in res.docs:
            doc = RagDocument.from_mapping(d.__dict__)
            try:
                score = float(getattr(d, "vector_score", 1.0))
            except (TypeError, ValueError):
                score = 1.0
            out.append(SearchResult(document=doc, score=score, key=getattr(d, "id", ""),
                                    origin="vector"))
        # Belt-and-braces: the TAG filter is mandatory above; re-check anyway.
        return check_candidates_match(out, candidate_id)

    def keyword_search(
        self,
        candidate_id: str,
        terms: List[str],
        k: Optional[int] = None,
        filters: Optional[SearchFilters] = None,
    ) -> List[SearchResult]:
        """Lexical FT.SEARCH over the TEXT field, scoped to `candidate_id`
        at the query level (never global-search-then-filter)."""
        if not candidate_id:
            raise ValueError("candidate_id is required (isolation).")
        from redis.commands.search.query import Query
        from redis.exceptions import ResponseError

        clean = [sanitize_text_term(t) for t in (terms or [])]
        clean = [t for t in clean if t]
        if not clean:
            return []
        k = k or self.config.top_k
        filters = filters or SearchFilters()
        base = f"@candidate_id:{{{escape_tag(candidate_id)}}}"
        union = "|".join(clean[:12])
        extra = self._filter_expr(filters)
        text_clause = f"@text:({union})"
        expr = f"({base} {text_clause} {extra})" if extra else f"({base} {text_clause})"
        query = Query(expr).paging(0, k)
        try:
            res = self._ft().search(query)
        except ResponseError as e:
            if "no such index" in str(e).lower():
                raise RagUnavailableError(
                    f"Index {self.config.index_name} does not exist: call ensure_index() first."
                )
            raise RagUnavailableError(f"keyword search failed: {e}")
        except Exception as e:
            raise RagUnavailableError(f"keyword search failed: {type(e).__name__}: {e}")

        # Redis returns best-first by TF-IDF; convert rank to a distance-like
        # score so all origins share the lower-is-better convention.
        out = []
        n = max(len(res.docs), 1)
        for i, d in enumerate(res.docs):
            doc = RagDocument.from_mapping(d.__dict__)
            out.append(SearchResult(document=doc, score=i / n,
                                    key=getattr(d, "id", ""), origin="keyword"))
        return check_candidates_match(out, candidate_id)


# ---------------------------------------------------------- module API -----
def ensure_index(
    dimension: Optional[int] = None,
    client=None,
    config: Optional[RagConfig] = None,
    provider=None,
) -> Dict[str, Any]:
    """Idempotent index bootstrap. Never raises for infrastructure states.

    Dimension resolution: explicit arg > RAG_EMBEDDING_DIM env >
    provider.dimension. Returns a status dict (see RedisVectorStore.ensure_index).
    """
    from .embeddings import get_embedding_provider

    config = config or RagConfig.from_env()
    if dimension is None:
        env_dim = os.getenv("RAG_EMBEDDING_DIM")
        if env_dim:
            try:
                dimension = int(env_dim)
            except ValueError:
                return {"ok": False, "status": "error",
                        "reason": f"RAG_EMBEDDING_DIM must be an integer, got {env_dim!r}."}
    if dimension is None:
        provider = provider or get_embedding_provider()
        dimension = provider.dimension
    store_mode = os.getenv("RAG_STORE_MODE", "").strip().lower()
    if store_mode == "memory":
        return get_shared_in_memory_store(dimension, config).ensure_index(dimension)
    if client is None:
        client = connect_redis(config)
    if client is None:
        return {
            "ok": False,
            "status": "unavailable",
            "reason": f"No Redis at {config.redis_host}:{config.redis_port}; "
                      "session/history still work, vector search disabled.",
        }
    return RedisVectorStore(client, config).ensure_index(dimension)


def get_vector_store(
    mode: Optional[str] = None,
    dimension: Optional[int] = None,
    client=None,
    config: Optional[RagConfig] = None,
    provider=None,
    shared: bool = True,
) -> VectorStore:
    """Backend selector. ``mode``: 'redis' | 'memory' | 'auto'.

    'auto' prefers real Redis Stack and falls back to the in-memory store
    (handy for local dev/tests); the chosen backend is always visible as
    ``store.backend`` so misconfiguration can't hide silently.
    """
    from .embeddings import get_embedding_provider

    config = config or RagConfig.from_env()
    if mode is None:
        mode = os.getenv("RAG_STORE_MODE", "auto")
    mode = (mode or "auto").strip().lower()
    if mode == "memory":
        if dimension is None:
            provider = provider or get_embedding_provider()
            dimension = provider.dimension
        if shared:
            return get_shared_in_memory_store(dimension, config)
        return InMemoryVectorStore(dimension, config)
    if mode in ("redis", "auto"):
        client = client or connect_redis(config)
        if client is not None:
            # Fail fast with guidance when Redis lacks the search module,
            # instead of failing obscurely on every FT.SEARCH call.
            if not has_vector_search(client, config.redis_host, config.redis_port):
                _warn_no_search_once(config.redis_host, config.redis_port)
                raise RagUnavailableError(
                    f"Redis at {config.redis_host}:{config.redis_port} has no RediSearch module "
                    "(plain Redis?). Vector search needs the redis/redis-stack image — "
                    "see docker-compose.yml."
                )
            if dimension is None:
                provider = provider or get_embedding_provider()
                dimension = provider.dimension
            return RedisVectorStore(client, config, dimension)
        if mode == "redis":
            raise RagUnavailableError(
                f"Redis unreachable at {config.redis_host}:{config.redis_port}."
            )
        if dimension is None:
            provider = provider or get_embedding_provider()
            dimension = provider.dimension
        if shared:
            return get_shared_in_memory_store(dimension, config)
        return InMemoryVectorStore(dimension, config)
    raise ValueError(f"Unknown vector store mode {mode!r}. Expected 'redis', 'memory' or 'auto'.")
