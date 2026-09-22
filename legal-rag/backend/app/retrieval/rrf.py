"""Reciprocal Rank Fusion (RRF).

WHY RRF over a weighted score sum: BM25 scores and cosine similarities live on
completely different, unnormalized scales, so adding or averaging them requires
hand-tuned, corpus-specific weights that break the moment the corpus changes.
RRF ignores raw scores and fuses on *rank position* alone:

    score(d) = sum over retrievers of 1 / (k + rank_r(d))

This is scale-free, needs no tuning, and is robust: a document ranked highly by
either retriever floats to the top, while documents both retrievers rank low
stay low. ``k`` (default 60, the value from the original Cormack et al. paper)
dampens the influence of very high ranks so a single #1 hit cannot completely
dominate the fused list.
"""

from __future__ import annotations

from ..schemas import Chunk, ScoredChunk


def reciprocal_rank_fusion(
    rankings: list[list[Chunk]], k: int = 60
) -> list[ScoredChunk]:
    """Fuse several ranked lists of chunks into one, deduped by chunk key.

    Args:
        rankings: one ranked ``list[Chunk]`` per retriever (e.g. BM25, vector).
            Rank 0 is best.
        k: RRF smoothing constant.

    Returns:
        Chunks sorted by descending fused score, each carrying that score.
    """

    scores: dict[str, float] = {}
    seen: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            key = chunk.key
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            # Keep the first-seen chunk object as the canonical representative.
            seen.setdefault(key, chunk)

    fused = [
        ScoredChunk(chunk=seen[key], score=score)
        for key, score in scores.items()
    ]
    fused.sort(key=lambda sc: sc.score, reverse=True)
    return fused
