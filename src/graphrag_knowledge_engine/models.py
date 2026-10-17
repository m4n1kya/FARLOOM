"""Pydantic models shared across ingestion, retrieval, answering, and the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ChunkStrategy = Literal["recursive", "heading"]
RetrievalMode = Literal["dense", "bm25", "hybrid", "hybrid_graph"]

RETRIEVAL_MODES: tuple[RetrievalMode, ...] = ("dense", "bm25", "hybrid", "hybrid_graph")


class Chunk(BaseModel):
    """A retrievable unit of text with provenance metadata."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    doc_id: str
    text: str
    source: str  # file name of the originating document
    page: int | None = None  # 1-based PDF page, None for text formats
    section: str | None = None  # heading path, e.g. "On-Call > Escalation"
    position: int  # 0-based order within the document
    strategy: ChunkStrategy


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    title: str
    fmt: Literal["pdf", "markdown", "text"]
    num_chunks: int


class Triple(BaseModel):
    """A (subject, relation, object) fact extracted from a chunk."""

    model_config = ConfigDict(extra="forbid")

    subject: str
    relation: str
    obj: str
    chunk_id: str
    doc_id: str


class TaggedContext(BaseModel):
    """A chunk handed to the LLM with its citation tag."""

    model_config = ConfigDict(extra="forbid")

    tag: int  # citation index, 1-based
    chunk_id: str
    text: str
    source: str
    section: str | None = None
    page: int | None = None


class LegHit(BaseModel):
    """One hit from a single retrieval leg, pre-fusion."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    rank: int  # 1-based rank within the leg
    score: float  # leg-native score (cosine sim, BM25, or graph score)


class FusedCandidate(BaseModel):
    """A chunk after Reciprocal Rank Fusion across legs."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    fused_score: float
    leg_ranks: dict[str, int] = Field(default_factory=dict)  # leg name -> 1-based rank


class RetrievalDebug(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: RetrievalMode
    legs: dict[str, list[LegHit]] = Field(default_factory=dict)
    fused: list[FusedCandidate] = Field(default_factory=list)
    query_entities: list[str] = Field(default_factory=list)
    max_fused_score: float = 0.0
    # Confidence in [0, 1]: max fused score divided by the mode's ceiling
    # (every leg putting the same chunk at rank 1). Comparable across modes,
    # unlike the raw RRF score which scales with the number of legs.
    confidence: float = 0.0


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tag: int
    chunk_id: str
    source: str
    section: str | None = None
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    refused: bool
    citations: list[Citation] = Field(default_factory=list)
    debug: RetrievalDebug


class IngestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    strategy: ChunkStrategy = "recursive"


class IngestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    num_chunks: int
    num_triples: int
    strategy: ChunkStrategy


class NeighborEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    relation: str
    obj: str
    chunk_ids: list[str]


class NeighborsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity: str
    found: bool
    edges: list[NeighborEdge] = Field(default_factory=list)


class EvalConfigResult(BaseModel):
    """Aggregate retrieval/answer metrics for one retrieval mode."""

    model_config = ConfigDict(extra="forbid")

    mode: RetrievalMode
    hit_at_k: float
    mrr: float
    context_precision: float
    faithfulness: float
    answered: int
    refused: int
    total: int


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int
    num_questions: int
    results: list[EvalConfigResult]


class EvalRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gold_path: str = "data/gold/qa_pairs.json"
    corpus_dir: str = "data/corpus"
    k: int = 5
