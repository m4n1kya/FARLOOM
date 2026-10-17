"""Sparse retrieval leg: BM25 (Okapi) over tokenized chunks.

The index is rebuilt from the SQLite chunk table whenever the corpus changes;
for corpora this engine targets (thousands of chunks) a rebuild is
milliseconds, which keeps the index trivially consistent with the store.
Hits with score <= 0 are dropped so unrelated queries yield an empty leg
instead of noise - that emptiness is what lets low-confidence refusal work.
"""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from graphrag_knowledge_engine.providers.embeddings import content_tokens


class Bm25Index:
    def __init__(self, corpus: list[tuple[str, str]]) -> None:
        """corpus: list of (chunk_id, text)."""
        self._ids = [chunk_id for chunk_id, _ in corpus]
        self._bm25: BM25Okapi | None = None
        if corpus:
            self._bm25 = BM25Okapi([content_tokens(text) for _, text in corpus])

    @property
    def size(self) -> int:
        return len(self._ids)

    def query(self, text: str, top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, bm25_score) pairs with score > 0, best first."""
        if self._bm25 is None or top_k <= 0:
            return []
        tokens = content_tokens(text)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            ((self._ids[i], float(s)) for i, s in enumerate(scores) if float(s) > 0.0),
            key=lambda item: (-item[1], item[0]),
        )
        return ranked[:top_k]
