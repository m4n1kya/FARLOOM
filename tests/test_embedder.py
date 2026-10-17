"""FakeEmbedder: determinism, shape, normalization, lexical signal."""

import math

import pytest

from graphrag_knowledge_engine.providers.embeddings import FakeEmbedder


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_deterministic_and_normalized() -> None:
    embedder = FakeEmbedder(dim=128)
    v1, v2 = embedder.embed_texts(["Beacon API rate limits", "Beacon API rate limits"])
    assert v1 == v2
    assert len(v1) == 128
    assert math.sqrt(sum(x * x for x in v1)) == pytest.approx(1.0)


def test_shared_vocabulary_is_closer() -> None:
    embedder = FakeEmbedder(dim=256)
    a, b, c = embedder.embed_texts(
        [
            "the beacon api enforces a rate limit",
            "beacon api rate limit per agent",
            "annual leave and public holidays policy",
        ]
    )
    assert _cosine(a, b) > _cosine(a, c)


def test_empty_text_is_zero_vector() -> None:
    (v,) = FakeEmbedder(dim=64).embed_texts([""])
    assert all(x == 0.0 for x in v)


def test_rejects_tiny_dimension() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dim=4)
