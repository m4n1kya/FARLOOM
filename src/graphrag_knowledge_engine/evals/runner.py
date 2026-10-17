"""Eval runner: scores every retrieval mode against the gold QA set.

For each mode (dense / bm25 / hybrid / hybrid_graph) the runner retrieves for
every gold question, marks retrieved chunks relevant when they match a gold
evidence spec (same source document AND containing the expected snippet),
and aggregates hit@k, MRR, context precision, and answer faithfulness.

The engine is built fresh in a scratch directory for every invocation so eval
results never depend on leftover state.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.engine import Engine, create_engine
from graphrag_knowledge_engine.evals.faithfulness import score_faithfulness
from graphrag_knowledge_engine.evals.metrics import (
    context_precision,
    hit_at_k,
    mean_reciprocal_rank,
)
from graphrag_knowledge_engine.models import (
    RETRIEVAL_MODES,
    Chunk,
    ChunkStrategy,
    EvalConfigResult,
    EvalReport,
    RetrievalMode,
)


class GoldEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc: str  # source file name
    must_contain: str  # snippet the relevant chunk must contain (casefolded)

    def matches(self, chunk: Chunk) -> bool:
        return chunk.source == self.doc and self.must_contain.casefold() in chunk.text.casefold()


class GoldQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    kind: str = "single-hop"  # informational: single-hop | multi-hop
    evidence: list[GoldEvidence]


class GoldDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[GoldQuestion]


def load_gold(path: Path) -> GoldDataset:
    return GoldDataset.model_validate(json.loads(path.read_text(encoding="utf-8")))


def run_eval(
    corpus_dir: Path,
    gold_path: Path,
    k: int = 5,
    strategy: ChunkStrategy = "heading",
    modes: tuple[RetrievalMode, ...] = RETRIEVAL_MODES,
    base_config: EngineConfig | None = None,
) -> EvalReport:
    gold = load_gold(gold_path)
    corpus_files = sorted(
        p for p in corpus_dir.iterdir() if p.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}
    )
    if not corpus_files:
        raise ValueError(f"No corpus documents found in {corpus_dir}")

    with tempfile.TemporaryDirectory(prefix="graphrag-eval-", ignore_cleanup_errors=True) as scratch:
        config = (base_config or EngineConfig()).model_copy(
            update={"data_dir": Path(scratch)}
        )
        engine = create_engine(config)
        try:
            for doc_path in corpus_files:
                engine.ingest_path(doc_path, strategy=strategy)
            results = [_eval_mode(engine, gold, mode, k) for mode in modes]
        finally:
            engine.close()

    return EvalReport(k=k, num_questions=len(gold.questions), results=results)


def _eval_mode(
    engine: Engine, gold: GoldDataset, mode: RetrievalMode, k: int
) -> EvalConfigResult:
    relevance_lists: list[list[bool]] = []
    hits = 0.0
    precisions: list[float] = []
    faith_scores: list[float] = []
    answered = 0
    refused = 0

    for question in gold.questions:
        response = engine.ask(question.question, mode=mode)
        retrieved_ids = [c.chunk_id for c in response.debug.fused]
        chunks = engine.store.get_chunks(retrieved_ids)
        relevance = [
            any(ev.matches(chunks[cid]) for ev in question.evidence) if cid in chunks else False
            for cid in retrieved_ids
        ]
        relevance_lists.append(relevance)
        hits += hit_at_k(relevance, k) if relevance else 0.0
        precisions.append(context_precision(relevance, k))
        if response.refused:
            refused += 1
        else:
            answered += 1
            faith_scores.append(score_faithfulness(response, engine.llm, engine.store))

    n = len(gold.questions)
    return EvalConfigResult(
        mode=mode,
        hit_at_k=hits / n if n else 0.0,
        mrr=mean_reciprocal_rank(relevance_lists),
        context_precision=sum(precisions) / n if n else 0.0,
        faithfulness=sum(faith_scores) / len(faith_scores) if faith_scores else 0.0,
        answered=answered,
        refused=refused,
        total=n,
    )


def format_table(report: EvalReport) -> str:
    """Render the comparison table printed by `make eval` (and in the README)."""
    headers = [
        "config",
        f"hit@{report.k}",
        "MRR",
        f"ctx_precision@{report.k}",
        "faithfulness",
        "answered",
    ]
    rows = [
        [
            r.mode,
            f"{r.hit_at_k:.3f}",
            f"{r.mrr:.3f}",
            f"{r.context_precision:.3f}",
            f"{r.faithfulness:.3f}",
            f"{r.answered}/{r.total}",
        ]
        for r in report.results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "-+-".join("-" * w for w in widths)
    lines = [
        " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)),
        sep,
        *(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows),
    ]
    return "\n".join(lines)
