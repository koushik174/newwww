"""Embedding providers with a graceful offline fallback.

WHY a mock embedder: the whole pipeline must run offline with zero downloads so
it can be demoed, tested and used in CI without a GPU or network. The
``MockEmbedder`` is a pure-Python, deterministic hashing embedder: it maps token
hashes into a fixed-dimensional vector and L2-normalizes, giving a stable
term-overlap signal that behaves enough like a dense retriever to exercise the
full graph. Set ``USE_REAL_EMBEDDINGS=1`` to swap in real
``sentence-transformers`` embeddings (all-MiniLM-L6-v2) behind the identical
``embed`` interface.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from .config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder(Protocol):
    """Common interface so the vector index never cares which backend runs."""

    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class MockEmbedder:
    """Deterministic, dependency-free embedder for offline operation.

    Each token is hashed into a small set of dimensions (a signed
    feature-hashing / hashing-trick vector). Documents that share vocabulary
    end up with high cosine similarity, which is sufficient to make dense
    retrieval meaningfully different from BM25 while staying fully reproducible.
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = _tokenize(text)
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).digest()
            idx = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """Real dense embeddings via sentence-transformers (lazy import).

    Kept behind the ``USE_REAL_EMBEDDINGS`` gate because loading the model
    downloads weights and needs torch. The interface matches ``MockEmbedder``
    exactly so nothing downstream changes.
    """

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [v.tolist() for v in vecs]


def get_embedder(settings: Settings) -> Embedder:
    """Factory: real embeddings when explicitly enabled, mock otherwise.

    Any failure to load the real model (missing package, no network) falls back
    to the mock rather than crashing, so a misconfigured env degrades to
    "works offline" instead of "500s on startup".
    """

    if settings.use_real_embeddings:
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception:  # pragma: no cover - depends on optional deps
            pass
    return MockEmbedder(dim=settings.embedding_dim)
