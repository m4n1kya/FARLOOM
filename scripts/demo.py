"""End-to-end offline demo.

Builds a fresh engine on the deterministic fake providers, ingests the
bundled corpus, asks single-hop and multi-hop questions, shows a graph
neighborhood and a low-confidence refusal, then runs the full eval matrix.
No network, no API keys, no services.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from graphrag_knowledge_engine.config import EngineConfig  # noqa: E402
from graphrag_knowledge_engine.engine import Engine, create_engine  # noqa: E402
from graphrag_knowledge_engine.evals.runner import format_table, run_eval  # noqa: E402
from graphrag_knowledge_engine.models import AskResponse  # noqa: E402


def show(response: AskResponse, engine: Engine | None = None, fused_rows: int = 0) -> None:
    print(f"\nQ: {response.question}")
    print(f"A: {response.answer}")
    for c in response.citations:
        where = c.source + (f" :: {c.section}" if c.section else "")
        print(f"   [{c.tag}] {where}")
    if response.debug.query_entities:
        print(f"   graph entities: {', '.join(response.debug.query_entities)}")
    print(f"   max fused score: {response.debug.max_fused_score:.4f}")
    if engine is None or fused_rows == 0:
        return
    chunks = engine.store.get_chunks([c.chunk_id for c in response.debug.fused[:fused_rows]])
    for i, cand in enumerate(response.debug.fused[:fused_rows]):
        legs = ", ".join(f"{leg}#{rank}" for leg, rank in sorted(cand.leg_ranks.items()))
        chunk = chunks.get(cand.chunk_id)
        where = f"{chunk.source} :: {chunk.section}" if chunk else cand.chunk_id
        print(f"   fused#{i + 1} rrf={cand.fused_score:.4f} ({legs})  {where}")


def main() -> int:
    data_dir = ROOT / "artifacts" / "demo"
    shutil.rmtree(data_dir, ignore_errors=True)
    config = EngineConfig(data_dir=data_dir, provider="fake")
    engine = create_engine(config)

    print("=== Ingest (heading-aware chunking) ===")
    for doc in sorted((ROOT / "data" / "corpus").glob("*.md")):
        result = engine.ingest_path(doc, strategy="heading")
        print(
            f"  {result.source}: {result.num_chunks} chunks, "
            f"{result.num_triples} triples"
        )
    print(
        f"  knowledge graph: {engine.graph.num_nodes} entities, "
        f"{engine.graph.num_edges} edges"
    )

    print("\n=== Single-hop question ===")
    show(engine.ask("What rate limit does Beacon API enforce per customer agent?"))

    print("\n=== Multi-hop question (evidence reachable through the graph) ===")
    show(
        engine.ask(
            "Which engineering organization is led from the campus "
            "hosting the primary Pulse Engine cluster?"
        ),
        engine=engine,
        fused_rows=3,
    )

    print("\n=== Low-confidence refusal ===")
    show(engine.ask("What is the boiling point of mercury on Venus?"))

    print("\n=== Graph neighborhood: 'Helios Sync' (depth 1) ===")
    for edge in engine.neighbors("Helios Sync", depth=1).edges[:8]:
        print(f"  ({edge.subject}) -[{edge.relation}]-> ({edge.obj})")

    engine.close()

    print("\n=== Eval: dense vs bm25 vs hybrid vs hybrid+graph ===")
    report = run_eval(
        ROOT / "data" / "corpus",
        ROOT / "data" / "gold" / "qa_pairs.json",
        k=5,
    )
    print(format_table(report))

    best = max(report.results, key=lambda r: (r.hit_at_k, r.mrr))
    print(f"\nBest config on this corpus: {best.mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
