"""Embedding service abstraction for the candidate RAG layer.

The rest of the application programs against :class:`EmbeddingProvider`
(``embed_text`` / ``embed_texts``) and never touches a vendor SDK directly,
so the provider/model can be swapped via environment variables.

Default provider uses the Gemini embedding API (same ecosystem and API key
as the existing interviewer). It deliberately does NOT use a generative
model: embeddings come from ``models/gemini-embedding-001`` (live-validated:
3072 native dims, supports ``output_dimensionality=768`` truncation).

Importing this module performs no network I/O.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import List, Optional

DEFAULT_MODEL = "models/gemini-embedding-001"
DEFAULT_DIMENSION = 768  # live-validated truncation of gemini-embedding-001

# Native (untruncated) dimensions, used to sanity-check RAG_EMBEDDING_DIM.
KNOWN_NATIVE_DIMS = {
    "models/gemini-embedding-001": 3072,
}

# Hard input guardrail; chunking (Phase 2) keeps inputs far below this.
MAX_INPUT_CHARS = 20000


class EmbeddingError(Exception):
    """Raised for provider misconfiguration or embedding failures."""


class EmbeddingProvider(ABC):
    """Vendor-neutral embedding interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider/model label, e.g. 'gemini:models/gemini-embedding-001'."""
        raise NotImplementedError

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Length of every vector this provider returns."""
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed one non-empty string. Raises on empty input or dim mismatch."""
        raise NotImplementedError

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch. Empty list returns []. Order is preserved."""
        raise NotImplementedError

    # -- shared validation helpers -------------------------------------
    def _check_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError(f"embed_text expects str, got {type(text).__name__}.")
        if not text.strip():
            raise ValueError("embed_text requires non-empty text.")
        if len(text) > MAX_INPUT_CHARS:
            raise EmbeddingError(
                f"Input too long ({len(text)} chars, max {MAX_INPUT_CHARS}). "
                "Split it into chunks before embedding."
            )
        return text

    def _check_vector(self, vector: List[float]) -> List[float]:
        vec = [float(v) for v in vector]
        if len(vec) != self.dimension:
            raise EmbeddingError(
                f"Provider {self.name} returned dim {len(vec)}, expected {self.dimension}."
            )
        return vec


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embedding API provider (NOT a generative model).

    Config (env or constructor args, args win):
      RAG_EMBEDDING_MODEL  default ``models/gemini-embedding-001``
      RAG_EMBEDDING_DIM    default 768 (validated truncation for the default model)
      GOOGLE_API_KEY       reused from the existing interviewer configuration
    """

    def __init__(
        self,
        model: Optional[str] = None,
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self._model = model or os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL)
        env_dim = os.getenv("RAG_EMBEDDING_DIM")
        if dimension is None and env_dim:
            try:
                dimension = int(env_dim)
            except ValueError:
                raise EmbeddingError(f"RAG_EMBEDDING_DIM must be an integer, got {env_dim!r}.")
        if dimension is None:
            if self._model in KNOWN_NATIVE_DIMS:
                dimension = DEFAULT_DIMENSION
            else:
                raise EmbeddingError(
                    f"Unknown embedding model {self._model!r}: set RAG_EMBEDDING_DIM explicitly."
                )
        if dimension <= 0:
            raise EmbeddingError(f"Embedding dimension must be positive, got {dimension}.")
        self._dimension = dimension
        self._api_key = api_key  # None -> read GOOGLE_API_KEY lazily per call

    @property
    def name(self) -> str:
        return f"gemini:{self._model}"

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model(self) -> str:
        return self._model

    def _key(self) -> str:
        key = self._api_key or os.getenv("GOOGLE_API_KEY") or ""
        if not key:
            raise EmbeddingError(
                "GOOGLE_API_KEY is not set. Add it to server/.env "
                "or pass api_key= explicitly."
            )
        return key

    def embed_text(self, text: str) -> List[float]:
        self._check_text(text)
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        for t in texts:
            self._check_text(t)
        try:
            import google.generativeai as genai

            genai.configure(api_key=self._key())
            resp = genai.embed_content(
                model=self._model,
                content=texts if len(texts) > 1 else texts[0],
                output_dimensionality=self._dimension,
            )
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Gemini embedding call failed: {e}")
        raw = resp["embedding"]
        # Single-string calls return one vector; batch calls return a list of vectors.
        vectors = raw if (isinstance(raw, list) and raw and isinstance(raw[0], list)) else [raw]
        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} vectors, got {len(vectors)}."
            )
        return [self._check_vector(v) for v in vectors]


class LocalEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding provider (offline, requires no API key).

    Generates stable pseudo-vectors based on SHA-256 hash expansion of input text,
    matching the configured RAG_EMBEDDING_DIM (default 768). Ideal for running
    RAG with Groq or in air-gapped/local environments.
    """

    def __init__(
        self,
        dimension: Optional[int] = None,
        **kwargs,
    ) -> None:
        env_dim = os.getenv("RAG_EMBEDDING_DIM")
        if dimension is None and env_dim:
            try:
                dimension = int(env_dim)
            except ValueError:
                raise EmbeddingError(f"RAG_EMBEDDING_DIM must be an integer, got {env_dim!r}.")
        self._dimension = dimension if dimension is not None else DEFAULT_DIMENSION
        if self._dimension <= 0:
            raise EmbeddingError(f"Embedding dimension must be positive, got {self._dimension}.")

    @property
    def name(self) -> str:
        return f"local:hash-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        self._check_text(text)
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        out = []
        for t in texts:
            self._check_text(t)
            buf = b""
            i = 0
            while len(buf) < self._dimension:
                buf += hashlib.sha256(t.encode("utf-8") + bytes([i % 256])).digest()
                i += 1
            out.append(self._check_vector([(b - 128) / 128.0 for b in buf[: self._dimension]]))
        return out


def get_embedding_provider(kind: Optional[str] = None, **kwargs) -> EmbeddingProvider:
    """Factory. ``kind`` defaults to env RAG_EMBEDDING_PROVIDER (or auto-detected)."""
    env_kind = os.getenv("RAG_EMBEDDING_PROVIDER")
    if kind is None:
        if env_kind:
            kind = env_kind.strip().lower()
        else:
            if os.getenv("GOOGLE_API_KEY"):
                kind = "gemini"
            elif os.getenv("GROQ_API_KEY"):
                kind = "local"
            else:
                kind = "gemini"
    else:
        kind = kind.strip().lower()

    if kind == "gemini":
        return GeminiEmbeddingProvider(**kwargs)
    if kind in ("local", "deterministic", "hash"):
        return LocalEmbeddingProvider(**kwargs)
    raise EmbeddingError(
        f"Unknown embedding provider {kind!r}. Supported: 'gemini', 'local'."
    )
