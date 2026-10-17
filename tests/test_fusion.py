"""RRF math against hand-computed fixtures."""

import pytest

from graphrag_knowledge_engine.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_hand_computed_two_legs() -> None:
    # dense: a, b, c        bm25: b, a, d
    # k = 60
    # a: 1/61 + 1/62          b: 1/62 + 1/61       (tie -> broken by id)
    # c: 1/63                 d: 1/63 (rank 3 in bm25)
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c"], "bm25": ["b", "a", "d"]}, k=60)
    by_id = {c.chunk_id: c.fused_score for c in fused}
    assert by_id["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert by_id["c"] == pytest.approx(1 / 63)
    assert by_id["d"] == pytest.approx(1 / 63)
    assert [c.chunk_id for c in fused] == ["a", "b", "c", "d"]  # a before b: id tiebreak


def test_rrf_three_legs_and_ranks() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": ["x", "y"], "bm25": ["y"], "graph": ["z", "y"]}, k=1
    )
    by_id = {c.chunk_id: c for c in fused}
    # y: 1/(1+2) + 1/(1+1) + 1/(1+2) = 1/3 + 1/2 + 1/3 = 7/6
    assert by_id["y"].fused_score == pytest.approx(7 / 6)
    assert by_id["x"].fused_score == pytest.approx(1 / 2)
    assert by_id["z"].fused_score == pytest.approx(1 / 2)
    assert by_id["y"].leg_ranks == {"dense": 2, "bm25": 1, "graph": 2}
    assert fused[0].chunk_id == "y"


def test_rrf_single_leg_preserves_order() -> None:
    fused = reciprocal_rank_fusion({"bm25": ["c1", "c2", "c3"]}, k=60)
    assert [c.chunk_id for c in fused] == ["c1", "c2", "c3"]


def test_rrf_empty_and_invalid_k() -> None:
    assert reciprocal_rank_fusion({}, k=60) == []
    assert reciprocal_rank_fusion({"dense": []}, k=60) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["a"]}, k=0)
