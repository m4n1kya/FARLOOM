"""Retrieval metrics over per-query relevance vectors.

Each query's retrieval result is reduced to a list of booleans: position i is
True when the i-th retrieved chunk is relevant per the gold annotations.
All metrics are pure functions over those vectors so they are trivially
unit-testable against hand-computed values.
"""

from __future__ import annotations

from collections.abc import Sequence


def hit_at_k(relevance: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-k retrieved items is relevant, else 0.0."""
    if k < 1:
        raise ValueError("k must be >= 1")
    return 1.0 if any(relevance[:k]) else 0.0


def reciprocal_rank(relevance: Sequence[bool]) -> float:
    """1/rank of the first relevant item, 0.0 when none is relevant."""
    for index, relevant in enumerate(relevance):
        if relevant:
            return 1.0 / (index + 1)
    return 0.0


def mean_reciprocal_rank(relevance_lists: Sequence[Sequence[bool]]) -> float:
    if not relevance_lists:
        return 0.0
    return sum(reciprocal_rank(r) for r in relevance_lists) / len(relevance_lists)


def context_precision(relevance: Sequence[bool], k: int) -> float:
    """Fraction of the top-k retrieved items that are relevant.

    Computed over min(k, retrieved) items; 0.0 when nothing was retrieved.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    window = relevance[:k]
    if not window:
        return 0.0
    return sum(1.0 for r in window if r) / len(window)
