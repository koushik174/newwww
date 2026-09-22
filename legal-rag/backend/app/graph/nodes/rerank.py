"""Node 4: cross-encoder reranking to Top-K.

WHY here and not earlier: reranking is the precision stage. Retrieval + RRF
optimize recall (get the right passage *somewhere* in the top ~40); the
cross-encoder then reads each (query, passage) pair jointly and reorders them by
true relevance, and we keep only ``RERANK_TOP_K``. Trimming to a small, highly
relevant set is what keeps the generation context focused and cheap, and is
what makes uncited/hallucinated claims easy to catch downstream.
"""

from __future__ import annotations

from ...config import Settings
from ...retrieval.reranker import Reranker
from ..state import GraphState


def rerank(state: GraphState) -> GraphState:
    deps = state["_deps"]
    reranker: Reranker = deps["reranker"]
    settings: Settings = deps["settings"]
    query = state.get("sanitized_query") or state.get("query", "")
    top_k = state.get("top_k") or settings.rerank_top_k

    candidates = state.get("candidates", [])
    reranked = reranker.rerank(query, candidates, top_k)
    return {"reranked": reranked}
