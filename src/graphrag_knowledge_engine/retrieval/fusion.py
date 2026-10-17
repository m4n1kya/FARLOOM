"""Reciprocal Rank Fusion (RRF).

For document ``d`` and result lists ``L`` (one per retrieval leg):

    RRF(d) = sum over legs l of  1 / (k + rank_l(d))

where ``rank_l(d)`` is d's 1-based rank in leg ``l`` (legs that did not
retrieve ``d`` contribute nothing) and ``k`` is a smoothing constant
(k = 60, following Cormack et al., 2009).

RRF is used instead of score normalization because the legs' scores live on
incompatible scales (cosine similarity in [-1, 1], unbounded BM25, graph hop
weights). Ranks are scale-free, so no per-leg calibration is needed and a
single outlier score cannot dominate the fusion.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from graphrag_knowledge_engine.models import FusedCandidate


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    k: int = 60,
) -> list[FusedCandidate]:
    """Fuse per-leg ranked id lists into a single ranking.

    Args:
        rankings: leg name -> ids ordered best-first (rank 1 = index 0).
        k: smoothing constant; larger k flattens the rank contribution curve.

    Returns:
        Candidates sorted by fused score (desc), ties broken by id for
        determinism. Each candidate records its per-leg ranks for debugging.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    scores: dict[str, float] = {}
    leg_ranks: dict[str, dict[str, int]] = {}
    for leg, ids in rankings.items():
        for index, chunk_id in enumerate(ids):
            rank = index + 1
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            leg_ranks.setdefault(chunk_id, {})[leg] = rank
    fused = [
        FusedCandidate(chunk_id=cid, fused_score=score, leg_ranks=leg_ranks[cid])
        for cid, score in scores.items()
    ]
    fused.sort(key=lambda c: (-c.fused_score, c.chunk_id))
    return fused
