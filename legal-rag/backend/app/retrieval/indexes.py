"""Per-corpus retrieval indexes: BM25 (lexical) and Vector (dense).

WHY hybrid retrieval: legal queries mix precise lexical tokens that MUST match
exactly ("42 U.S.C. 1983", "180 days", a specific section number) with
semantic intent ("can I sue a city for a constitutional violation?"). BM25
nails the former and is unbeatable on rare exact terms; dense vectors capture
the latter, matching paraphrases the query never spells out. Running both and
fusing (see rrf.py) gives recall neither achieves alone.

Both index types expose the SAME interface::

    index.search(query: str, top_n: int) -> list[Chunk]

so the retrieve node, the fusion step and the optional Qdrant backend are all
interchangeable. rank-bm25 and numpy are used when installed but each has a
pure-Python fallback so the system runs with zero third-party packages.
"""

from __future__ import annotations

import math
import re
from typing import Optional, Protocol

from ..config import Settings
from ..embeddings import Embedder, get_embedder
from ..schemas import Chunk, SourceType

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class SearchIndex(Protocol):
    def search(self, query: str, top_n: int) -> list[Chunk]:
        ...


# --------------------------------------------------------------------------- #
# BM25 (lexical)
# --------------------------------------------------------------------------- #
class _PurePythonBM25:
    """Minimal Okapi BM25 used when ``rank-bm25`` is not installed.

    Implements the standard BM25 scoring so the lexical retriever always works,
    even in a bare stdlib environment. ``rank-bm25`` is preferred when present
    because it is battle-tested and faster on large corpora.
    """

    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_tokens = corpus_tokens
        self.doc_len = [len(d) for d in corpus_tokens]
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if corpus_tokens else 0.0
        self.n_docs = len(corpus_tokens)
        self.df: dict[str, int] = {}
        for doc in corpus_tokens:
            for term in set(doc):
                self.df[term] = self.df.get(term, 0) + 1
        self.tf: list[dict[str, int]] = []
        for doc in corpus_tokens:
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            self.tf.append(counts)

    def _idf(self, term: str) -> float:
        n_q = self.df.get(term, 0)
        # BM25+ style idf, floored at ~0 to avoid negatives.
        return math.log(1 + (self.n_docs - n_q + 0.5) / (n_q + 0.5))

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.n_docs
        for term in query_tokens:
            if term not in self.df:
                continue
            idf = self._idf(term)
            for i in range(self.n_docs):
                f = self.tf[i].get(term, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (
                    1 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1)
                )
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores


class BM25Index:
    """Lexical retriever over a single corpus."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._corpus_tokens = [_tokenize(c.text + " " + c.title) for c in chunks]
        self._backend = self._build_backend()

    def _build_backend(self):
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            return ("rank_bm25", BM25Okapi(self._corpus_tokens))
        except Exception:
            return ("pure", _PurePythonBM25(self._corpus_tokens))

    def search(self, query: str, top_n: int) -> list[Chunk]:
        if not self.chunks:
            return []
        q_tokens = _tokenize(query)
        kind, bm = self._backend
        scores = list(bm.get_scores(q_tokens))
        ranked = sorted(
            range(len(self.chunks)), key=lambda i: scores[i], reverse=True
        )
        return [self.chunks[i] for i in ranked[:top_n]]


# --------------------------------------------------------------------------- #
# Vector (dense) — numpy or pure-python, plus optional Qdrant
# --------------------------------------------------------------------------- #
def _cosine(a: list[float], b: list[float]) -> float:
    # Embedders here L2-normalize, so dot product == cosine similarity.
    return sum(x * y for x, y in zip(a, b))


class NumpyVectorIndex:
    """In-memory dense index. Uses numpy when available, else pure python.

    This is the default backend: no external service, instant startup, ideal
    for the seed corpus and CI. Swap to :class:`QdrantVectorIndex` for
    document-scale corpora by setting ``VECTOR_BACKEND=qdrant``.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder) -> None:
        self.chunks = chunks
        self.embedder = embedder
        texts = [c.text for c in chunks]
        self._matrix = embedder.embed(texts) if texts else []
        try:
            import numpy as np  # type: ignore

            self._np = np
            self._np_matrix = np.array(self._matrix, dtype="float32") if self._matrix else None
        except Exception:
            self._np = None
            self._np_matrix = None

    def search(self, query: str, top_n: int) -> list[Chunk]:
        if not self.chunks:
            return []
        q = self.embedder.embed([query])[0]
        if self._np is not None and self._np_matrix is not None:
            qv = self._np.array(q, dtype="float32")
            sims = self._np_matrix @ qv
            order = self._np.argsort(-sims)[:top_n]
            return [self.chunks[int(i)] for i in order]
        sims = [_cosine(q, row) for row in self._matrix]
        ranked = sorted(range(len(self.chunks)), key=lambda i: sims[i], reverse=True)
        return [self.chunks[i] for i in ranked[:top_n]]


class QdrantVectorIndex:
    """Optional Qdrant-backed dense index (same interface as the numpy one).

    Kept import-lazy and behind ``VECTOR_BACKEND=qdrant`` so the dependency is
    never required for offline use. This is the production swap for scaling
    beyond what fits in memory; docker-compose.yml provisions the service.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        settings: Settings,
        collection: str,
    ) -> None:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.models import Distance, PointStruct, VectorParams  # type: ignore

        self.chunks = chunks
        self.embedder = embedder
        self.collection = collection
        self._client = QdrantClient(url=settings.qdrant_url)
        vectors = embedder.embed([c.text for c in chunks]) if chunks else []
        dim = len(vectors[0]) if vectors else embedder.dim
        self._client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        if vectors:
            self._client.upsert(
                collection_name=collection,
                points=[
                    PointStruct(id=i, vector=vectors[i], payload={"idx": i})
                    for i in range(len(vectors))
                ],
            )

    def search(self, query: str, top_n: int) -> list[Chunk]:
        if not self.chunks:
            return []
        qv = self.embedder.embed([query])[0]
        hits = self._client.search(
            collection_name=self.collection, query_vector=qv, limit=top_n
        )
        return [self.chunks[int(h.payload["idx"])] for h in hits]


def build_vector_index(
    chunks: list[Chunk], embedder: Embedder, settings: Settings, name: str
) -> SearchIndex:
    """Choose the configured vector backend, falling back to numpy on error."""
    if settings.vector_backend == "qdrant":
        try:
            return QdrantVectorIndex(
                chunks,
                embedder,
                settings,
                collection=f"{settings.qdrant_collection_prefix}_{name}",
            )
        except Exception:  # pragma: no cover - needs a running Qdrant
            pass
    return NumpyVectorIndex(chunks, embedder)


# --------------------------------------------------------------------------- #
# Corpus store: bundles both indexes per source type
# --------------------------------------------------------------------------- #
class CorpusIndex:
    """Bundles the BM25 and Vector indexes for a single corpus."""

    def __init__(self, source_type: SourceType, chunks: list[Chunk], bm25: BM25Index, vector: SearchIndex):
        self.source_type = source_type
        self.chunks = chunks
        self.bm25 = bm25
        self.vector = vector


class CorpusStore:
    """Holds one :class:`CorpusIndex` per source type.

    Built once at startup (see ingest.build_indexes) and shared read-only by
    every request, so per-query latency is just search + fuse + rerank.
    """

    def __init__(self, settings: Settings, embedder: Optional[Embedder] = None) -> None:
        self.settings = settings
        self.embedder = embedder or get_embedder(settings)
        self.corpora: dict[SourceType, CorpusIndex] = {}

    def add_corpus(self, source_type: SourceType, chunks: list[Chunk]) -> None:
        bm25 = BM25Index(chunks)
        vector = build_vector_index(
            chunks, self.embedder, self.settings, source_type.value
        )
        self.corpora[source_type] = CorpusIndex(source_type, chunks, bm25, vector)

    def counts(self) -> dict[str, int]:
        return {st.value: len(ci.chunks) for st, ci in self.corpora.items()}
