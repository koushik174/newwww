"""Cross-encoder reranker.

WHY a separate cross-encoder stage: BM25 and dense retrieval are *bi-encoders*
— query and document are embedded independently, so they trade precision for
speed and scale. A cross-encoder jointly encodes the (query, passage) pair and
scores their actual interaction, which is far more accurate at judging true
relevance but too expensive to run over the whole corpus. The standard pattern
is therefore: cheap hybrid retrieval to get a few dozen candidates, then an
expensive cross-encoder to reorder just those and keep the top-K. This is where
most of the end-to-end quality comes from.

Offline, a deterministic lexical-overlap reranker stands in for the real model
so the pipeline runs with zero downloads; set ``USE_REAL_RERANKER=1`` to use
``cross-encoder/ms-marco-MiniLM-L-6-v2`` behind the identical interface.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..config import Settings
from ..schemas import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        ...


class MockCrossEncoder:
    """Deterministic reranker: weighted token overlap (Jaccard + coverage).

    Not a real cross-encoder, but it produces a stable, query-dependent
    reordering good enough to exercise and test the pipeline offline. It
    rewards passages that cover more of the query's terms, which mirrors the
    behavior a real reranker exhibits on keyword-heavy legal queries.
    """

    def _score(self, query: str, chunk: Chunk) -> float:
        q = _tokens(query)
        if not q:
            return 0.0
        d = _tokens(chunk.text + " " + chunk.title)
        overlap = q & d
        coverage = len(overlap) / len(q)
        jaccard = len(overlap) / len(q | d) if (q | d) else 0.0
        return 0.7 * coverage + 0.3 * jaccard

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        scored = sorted(chunks, key=lambda c: self._score(query, c), reverse=True)
        return scored[:top_k]


class SentenceTransformerCrossEncoder:
    """Real cross-encoder reranker (lazy import, gated)."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder  # type: ignore

        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs)
        order = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
        return [chunks[i] for i in order[:top_k]]


def get_reranker(settings: Settings) -> Reranker:
    if settings.use_real_reranker:
        try:
            return SentenceTransformerCrossEncoder(settings.reranker_model)
        except Exception:  # pragma: no cover - optional dep
            pass
    return MockCrossEncoder()
