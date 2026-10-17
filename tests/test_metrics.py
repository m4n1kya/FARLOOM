"""hit@k / MRR / context precision against hand-computed values."""

import pytest

from graphrag_knowledge_engine.evals.metrics import (
    context_precision,
    hit_at_k,
    mean_reciprocal_rank,
    reciprocal_rank,
)


def test_hit_at_k() -> None:
    assert hit_at_k([False, False, True], k=3) == 1.0
    assert hit_at_k([False, False, True], k=2) == 0.0
    assert hit_at_k([], k=5) == 0.0
    with pytest.raises(ValueError):
        hit_at_k([True], k=0)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([False, True, True]) == pytest.approx(0.5)
    assert reciprocal_rank([True]) == 1.0
    assert reciprocal_rank([False, False]) == 0.0


def test_mean_reciprocal_rank_hand_computed() -> None:
    lists = [
        [True, False],          # rr = 1
        [False, False, True],   # rr = 1/3
        [False, False],         # rr = 0
    ]
    assert mean_reciprocal_rank(lists) == pytest.approx((1 + 1 / 3 + 0) / 3)
    assert mean_reciprocal_rank([]) == 0.0


def test_context_precision() -> None:
    assert context_precision([True, False, True, False], k=4) == pytest.approx(0.5)
    assert context_precision([True, True], k=5) == 1.0  # window = retrieved count
    assert context_precision([], k=5) == 0.0
    with pytest.raises(ValueError):
        context_precision([True], k=0)
