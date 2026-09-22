"""Tests for Reciprocal Rank Fusion ranking behavior."""

from __future__ import annotations

from app.retrieval.rrf import reciprocal_rank_fusion
from app.schemas import Chunk, SourceType


def _chunk(sid: str) -> Chunk:
    return Chunk(
        source_id=sid,
        source_type=SourceType.FEDERAL,
        title=sid,
        text=f"text for {sid}",
        paragraph="a",
    )


def test_rrf_rewards_agreement_across_retrievers():
    a, b, c = _chunk("a"), _chunk("b"), _chunk("c")
    # 'a' is top of both lists -> should win.
    bm25 = [a, b, c]
    vector = [a, c, b]
    fused = reciprocal_rank_fusion([bm25, vector], k=60)
    assert fused[0].chunk.source_id == "a"
    # scores strictly descending
    scores = [sc.score for sc in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_dedupes_by_chunk_key():
    a1 = _chunk("a")
    a2 = _chunk("a")  # same key
    b = _chunk("b")
    fused = reciprocal_rank_fusion([[a1, b], [a2]], k=60)
    keys = [sc.chunk.key for sc in fused]
    assert keys.count("a#a") == 1


def test_rrf_rank_based_not_score_based():
    """A doc ranked #1 by one retriever beats a doc ranked #2 by both."""
    x, y, z = _chunk("x"), _chunk("y"), _chunk("z")
    r1 = [x, y]      # x rank0, y rank1
    r2 = [z, y]      # z rank0, y rank1
    fused = {sc.chunk.source_id: sc.score for sc in reciprocal_rank_fusion([r1, r2], k=60)}
    # y appears at rank1 in both -> 2 * 1/(60+2)
    # x appears at rank0 once -> 1/(60+1)
    assert fused["y"] > fused["x"]
    assert fused["y"] > fused["z"]


def test_rrf_k_dampens_top_rank_dominance():
    a, b = _chunk("a"), _chunk("b")
    small_k = reciprocal_rank_fusion([[a], [b]], k=1)
    large_k = reciprocal_rank_fusion([[a], [b]], k=1000)
    # With larger k, the gap between rank0 contributions shrinks.
    small_scores = sorted(sc.score for sc in small_k)
    large_scores = sorted(sc.score for sc in large_k)
    assert (small_scores[-1] - small_scores[0]) >= (large_scores[-1] - large_scores[0])
