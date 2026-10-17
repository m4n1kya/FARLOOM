"""Embedding provider boundary.

Everything that needs embeddings depends on the ``EmbeddingProvider`` protocol,
never on a vendor SDK. Two implementations ship:

* ``OpenAIEmbedder`` - calls the OpenAI embeddings REST API via httpx.
* ``FakeEmbedder``  - deterministic feature-hashing embedder. Each content
  token (stopwords removed) is hashed (SHA-256) into one of ``dim`` buckets
  with a hash-derived sign, the bucket counts are accumulated, and the vector
  is L2-normalised. Tokens of four or more characters also contribute
  boundary-marked character trigrams at a lower weight, giving the dense leg
  the subword fuzziness real embedding models have ("rotation" lands near
  "rotated") while BM25 stays strictly exact-match - so the two legs fail
  differently, which is exactly what Reciprocal Rank Fusion exploits.
  Unrelated texts stay near-orthogonal because function words never
  contribute.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

import httpx

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Shared stopword list for the lexical pipeline (fake embedder, BM25, fake
# LLM scoring). Dropping these keeps function words from dominating either
# leg: under feature hashing they would otherwise make any two English texts
# look similar, and under BM25 they would let "what is the ..." match
# everything.
STOPWORDS = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "in", "on",
        "at", "to", "for", "and", "or", "by", "with", "as", "it", "its", "this", "that",
        "what", "which", "who", "how", "when", "where", "why", "does", "do", "did",
        "from", "their", "they", "we", "our", "your", "all", "any", "has", "have", "had",
        "will", "must", "may", "can", "not", "no", "than", "then", "there", "these",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokenizer shared by the fake embedder and BM25."""
    return _TOKEN_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    """Tokens minus stopwords - the unit indexed by the lexical legs."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


class EmbeddingProvider(Protocol):
    """Anything that can turn texts into fixed-dimension vectors."""

    @property
    def dimension(self) -> int: ...

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic, offline, dependency-free embedder (feature hashing)."""

    _TRIGRAM_WEIGHT = 0.3

    def __init__(self, dim: int = 512) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _features(self, token: str) -> list[tuple[str, float]]:
        feats = [(token, 1.0)]
        if len(token) >= 4:
            padded = f"^{token}$"
            feats.extend(
                (padded[i : i + 3], self._TRIGRAM_WEIGHT) for i in range(len(padded) - 2)
            )
        return feats

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in content_tokens(text):
            for feature, weight in self._features(token):
                digest = hashlib.sha256(feature.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self._dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[bucket] += sign * weight
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return vec
        return [v / norm for v in vec]


class OpenAIEmbedder:
    """Real embedder backed by the OpenAI embeddings endpoint."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dimension: int = 1536,
        timeout: float = 30.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAIEmbedder requires an API key")
        self._model = model
        self._dim = dimension
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._client.post(
            "/embeddings",
            json={"model": self._model, "input": list(texts), "dimensions": self._dim},
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        ordered = sorted(data, key=lambda item: int(item["index"]))
        return [[float(x) for x in item["embedding"]] for item in ordered]
