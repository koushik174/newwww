"""Node 3: per-corpus hybrid retrieval + RRF fusion.

WHY this shape: for each routed corpus we run BM25 (lexical) and the vector
index (dense) independently, then fuse their ranked lists with Reciprocal Rank
Fusion. Doing this *per corpus* (rather than pooling all documents first) keeps
each corpus's authority intact and lets the merge node reason about
statute-vs-regulation-vs-case ordering afterward. The output is a flat
candidate list plus a per-corpus breakdown, both carried in state for
observability (the demo prints candidate counts).
"""

from __future__ import annotations

from ...config import Settings
from ...retrieval.indexes import CorpusStore
from ...retrieval.rrf import reciprocal_rank_fusion
from ...schemas import Chunk
from ..state import GraphState


def retrieve(state: GraphState) -> GraphState:
    deps = state["_deps"]
    store: CorpusStore = deps["store"]
    settings: Settings = deps["settings"]
    query = state.get("sanitized_query") or state.get("query", "")

    per_corpus: dict[str, list[Chunk]] = {}
    all_candidates: list[Chunk] = []
    seen_keys: set[str] = set()

    for source_type in state.get("routed_to", []):
        corpus = store.corpora.get(source_type)
        if corpus is None:
            continue
        bm25_hits = corpus.bm25.search(query, settings.bm25_top_n)
        vector_hits = corpus.vector.search(query, settings.vector_top_n)
        fused = reciprocal_rank_fusion(
            [bm25_hits, vector_hits], k=settings.rrf_k
        )
        fused_chunks = [sc.chunk for sc in fused]
        per_corpus[source_type.value] = fused_chunks
        for chunk in fused_chunks:
            if chunk.key not in seen_keys:
                seen_keys.add(chunk.key)
                all_candidates.append(chunk)

    return {"retrieved": per_corpus, "candidates": all_candidates}
