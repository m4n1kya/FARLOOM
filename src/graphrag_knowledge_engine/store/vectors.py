"""ChromaDB persistent vector index.

ChromaDB's built-in embedding function is never used: it would download an
ONNX model on first call, which breaks the offline guarantee. Instead, every
``add`` and ``query`` passes embeddings produced by the configured
``EmbeddingProvider`` explicitly. Telemetry is disabled for the same reason.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.config import Settings


class VectorIndex:
    def __init__(self, persist_dir: Path, collection: str = "chunks") -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunk_ids: list[str], embeddings: list[list[float]]) -> None:
        if not chunk_ids:
            return
        # Embeddings are supplied explicitly - Chroma's default embedder never runs.
        self._collection.upsert(ids=chunk_ids, embeddings=cast(Any, embeddings))

    def query(self, embedding: list[float], top_k: int) -> list[tuple[str, float]]:
        """Return (chunk_id, cosine_similarity) pairs, best first."""
        if self.count() == 0 or top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=cast(Any, [embedding]),
            n_results=min(top_k, self.count()),
            include=["distances"],
        )
        ids = result["ids"][0]
        distances = (result.get("distances") or [[]])[0]
        # Chroma returns cosine *distance*; similarity = 1 - distance.
        return [(cid, 1.0 - float(dist)) for cid, dist in zip(ids, distances, strict=True)]

    def count(self) -> int:
        return int(self._collection.count())
