"""Hybrid retriever: runs the configured legs and fuses them with RRF.

Legs per mode:

* ``dense``        - ChromaDB cosine over provider embeddings
* ``bm25``         - Okapi BM25 over tokenized chunks
* ``hybrid``       - dense + bm25
* ``hybrid_graph`` - dense + bm25 + bounded knowledge-graph expansion

Each leg applies its own floor (cosine similarity floor for dense, score > 0
for BM25) before fusion, so a query with no real support produces an empty
candidate set and the answerer can refuse instead of hallucinating.
"""

from __future__ import annotations

from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.models import LegHit, RetrievalDebug, RetrievalMode
from graphrag_knowledge_engine.providers.embeddings import EmbeddingProvider
from graphrag_knowledge_engine.retrieval.bm25 import Bm25Index
from graphrag_knowledge_engine.retrieval.fusion import reciprocal_rank_fusion
from graphrag_knowledge_engine.store.graph import KnowledgeGraph
from graphrag_knowledge_engine.store.vectors import VectorIndex

_MODE_LEGS: dict[RetrievalMode, frozenset[str]] = {
    "dense": frozenset({"dense"}),
    "bm25": frozenset({"bm25"}),
    "hybrid": frozenset({"dense", "bm25"}),
    "hybrid_graph": frozenset({"dense", "bm25", "graph"}),
}


class HybridRetriever:
    def __init__(
        self,
        config: EngineConfig,
        embedder: EmbeddingProvider,
        vectors: VectorIndex,
        bm25: Bm25Index,
        graph: KnowledgeGraph,
    ) -> None:
        self._config = config
        self._embedder = embedder
        self._vectors = vectors
        self._bm25 = bm25
        self._graph = graph

    def retrieve(self, query: str, mode: RetrievalMode = "hybrid_graph") -> RetrievalDebug:
        cfg = self._config
        legs = _MODE_LEGS[mode]
        leg_hits: dict[str, list[LegHit]] = {}
        query_entities: list[str] = []

        if "dense" in legs:
            embedding = self._embedder.embed_texts([query])[0]
            dense = [
                (cid, sim)
                for cid, sim in self._vectors.query(embedding, cfg.dense_top_k)
                if sim >= cfg.dense_sim_floor
            ]
            leg_hits["dense"] = _to_hits(dense)

        if "bm25" in legs:
            leg_hits["bm25"] = _to_hits(self._bm25.query(query, cfg.bm25_top_k))

        if "graph" in legs:
            query_entities = self._graph.detect_query_entities(query)
            expanded = self._graph.expand(
                query_entities,
                max_depth=cfg.graph_depth,
                max_nodes=cfg.graph_max_nodes,
                top_k=cfg.graph_top_k,
            )
            leg_hits["graph"] = _to_hits(expanded)

        fused = reciprocal_rank_fusion(
            {leg: [h.chunk_id for h in hits] for leg, hits in leg_hits.items()},
            k=cfg.rrf_k,
        )
        max_fused = fused[0].fused_score if fused else 0.0
        # Normalize against the mode's ceiling: all legs agreeing at rank 1
        # scores len(legs) / (k + 1). This makes confidence comparable across
        # modes - a lone dense hit at rank 1 is full confidence in dense-only
        # mode but only one third of it when three legs were consulted.
        ceiling = len(legs) / (cfg.rrf_k + 1)
        return RetrievalDebug(
            mode=mode,
            legs=leg_hits,
            fused=fused,
            query_entities=query_entities,
            max_fused_score=max_fused,
            confidence=max_fused / ceiling if ceiling > 0 else 0.0,
        )


def _to_hits(scored: list[tuple[str, float]]) -> list[LegHit]:
    return [
        LegHit(chunk_id=cid, rank=i + 1, score=score) for i, (cid, score) in enumerate(scored)
    ]
