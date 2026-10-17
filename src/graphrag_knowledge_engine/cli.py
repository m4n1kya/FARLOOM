"""Command-line interface mirroring the HTTP API.

    graphrag ingest <path> [--strategy recursive|heading]
    graphrag ask "<question>" [--mode hybrid_graph]
    graphrag neighbors "<entity>" [--depth 1]
    graphrag eval [--corpus data/corpus] [--gold data/gold/qa_pairs.json] [--k 5]
    graphrag serve [--host 127.0.0.1] [--port 8000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from graphrag_knowledge_engine.config import config_from_env
from graphrag_knowledge_engine.models import AskResponse, ChunkStrategy, RetrievalMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="graphrag", description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None, help="Engine state directory")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest a PDF/Markdown/TXT file")
    p_ingest.add_argument("path", type=Path)
    p_ingest.add_argument(
        "--strategy", choices=["recursive", "heading"], default="recursive"
    )

    p_ask = sub.add_parser("ask", help="Ask a question over the indexed corpus")
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--mode",
        choices=["dense", "bm25", "hybrid", "hybrid_graph"],
        default="hybrid_graph",
    )
    p_ask.add_argument("--json", action="store_true", help="Print the full JSON response")

    p_nb = sub.add_parser("neighbors", help="Show knowledge-graph neighbors of an entity")
    p_nb.add_argument("entity")
    p_nb.add_argument("--depth", type=int, default=1)

    p_eval = sub.add_parser("eval", help="Run the retrieval eval across all modes")
    p_eval.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p_eval.add_argument("--gold", type=Path, default=Path("data/gold/qa_pairs.json"))
    p_eval.add_argument("--k", type=int, default=5)
    p_eval.add_argument(
        "--strategy", choices=["recursive", "heading"], default="heading"
    )

    p_serve = sub.add_parser("serve", help="Start the FastAPI server")
    p_serve.add_argument("--host", default=os.environ.get("GRAPHRAG_HOST", "127.0.0.1"))
    p_serve.add_argument("--port", type=int, default=int(os.environ.get("GRAPHRAG_PORT", "8000")))

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "serve":
        import uvicorn

        uvicorn.run("graphrag_knowledge_engine.api:app", host=args.host, port=args.port)
        return 0

    if args.command == "eval":
        from graphrag_knowledge_engine.evals.runner import format_table, run_eval

        report = run_eval(
            args.corpus,
            args.gold,
            k=args.k,
            strategy=cast(ChunkStrategy, args.strategy),
            base_config=config_from_env(args.data_dir),
        )
        print(f"Eval over {report.num_questions} questions (k={report.k}):\n")
        print(format_table(report))
        return 0

    from graphrag_knowledge_engine.engine import create_engine

    engine = create_engine(config_from_env(args.data_dir))
    try:
        if args.command == "ingest":
            result = engine.ingest_path(args.path, strategy=cast(ChunkStrategy, args.strategy))
            print(result.model_dump_json(indent=2))
        elif args.command == "ask":
            response = engine.ask(args.question, mode=cast(RetrievalMode, args.mode))
            if args.json:
                print(response.model_dump_json(indent=2))
            else:
                _print_answer(response)
        elif args.command == "neighbors":
            nb = engine.neighbors(args.entity, depth=args.depth)
            print(json.dumps(nb.model_dump(), indent=2))
    finally:
        engine.close()
    return 0


def _print_answer(response: AskResponse) -> None:
    print(f"Q: {response.question}")
    print(f"A: {response.answer}")
    if response.refused:
        print("(refused: low retrieval confidence)")
    for citation in response.citations:
        where = citation.source
        if citation.section:
            where += f" :: {citation.section}"
        if citation.page is not None:
            where += f" (p.{citation.page})"
        print(f"  [{citation.tag}] {where}")
    entities = response.debug.query_entities
    if entities:
        print(f"  graph entities: {', '.join(entities)}")
    print(f"  max fused score: {response.debug.max_fused_score:.4f}")


if __name__ == "__main__":
    sys.exit(main())
