"""FastAPI surface: /ingest, /ask, /graph/neighbors, /eval/run, /docs/list,
/graph/stats, /graph/entities, plus static frontend serving.

The engine is created once per process from environment config and shared via
``app.state``. ``create_app`` accepts an engine override so tests can inject
an engine built on fakes in a temp directory.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Additional response models for FARLOOM extensions
# ---------------------------------------------------------------------------
from pydantic import BaseModel, ConfigDict

from graphrag_knowledge_engine import __version__
from graphrag_knowledge_engine.config import config_from_env
from graphrag_knowledge_engine.engine import Engine, create_engine
from graphrag_knowledge_engine.evals.runner import run_eval
from graphrag_knowledge_engine.models import (
    AskResponse,
    Document,
    EvalReport,
    EvalRunRequest,
    IngestRequest,
    IngestResponse,
    NeighborsResponse,
    RetrievalMode,
)


class GraphStats(BaseModel):
    model_config = ConfigDict(extra="forbid")
    num_nodes: int
    num_edges: int
    entities: list[str]


class ProviderStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str
    demo_mode: bool
    version: str
    data_dir: str





def _auto_ingest_corpus(engine: Engine) -> None:
    """Ingest bundled corpora if the store is empty. Idempotent."""
    if engine.store.list_documents():
        return  # already populated (e.g. from a persisted data_dir)
    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    for subdir_name in ("corpus", "tech_corpus"):
        corpus_dir = base_dir / subdir_name
        if not corpus_dir.is_dir():
            corpus_dir = Path("data") / subdir_name
        if not corpus_dir.is_dir():
            continue
        for doc in sorted(corpus_dir.glob("*.md")):
            with suppress(Exception):
                engine.ingest_path(doc, strategy="heading")
        for doc in sorted(corpus_dir.glob("*.txt")):
            with suppress(Exception):
                engine.ingest_path(doc, strategy="recursive")
        for doc in sorted(corpus_dir.glob("*.pdf")):
            with suppress(Exception):
                engine.ingest_path(doc, strategy="recursive")


def get_engine(request: Request) -> Engine:
    eng: Engine = request.app.state.engine
    return eng


EngineDep = Annotated[Engine, Depends(get_engine)]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(engine: Engine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        eng = engine or create_engine(config_from_env())
        _auto_ingest_corpus(eng)
        app.state.engine = eng
        yield
        app.state.engine.close()

    app = FastAPI(
        title="FARLOOM — Technical Knowledge Intelligence Engine",
        version=__version__,
        description="Hybrid RAG: dense + BM25 + knowledge graph, fused with RRF.",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Original endpoints (preserved)
    # -----------------------------------------------------------------------

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(payload: IngestRequest, eng: EngineDep) -> IngestResponse:
        path = Path(payload.path)
        try:
            return eng.ingest_path(path, strategy=payload.strategy)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/ask", response_model=AskResponse)
    def ask(
        eng: EngineDep,
        q: Annotated[str, Query(min_length=1, description="The question to answer")],
        mode: Annotated[RetrievalMode, Query(description="Retrieval mode")] = "hybrid_graph",
    ) -> AskResponse:
        return eng.ask(q, mode=mode)

    @app.get("/graph/neighbors", response_model=NeighborsResponse)
    def neighbors(
        eng: EngineDep,
        entity: Annotated[str, Query(min_length=1)],
        depth: Annotated[int, Query(ge=1, le=3)] = 1,
    ) -> NeighborsResponse:
        return eng.neighbors(entity, depth=depth)

    @app.post("/eval/run", response_model=EvalReport)
    def eval_run(payload: EvalRunRequest, eng: EngineDep) -> EvalReport:
        corpus_dir = Path(payload.corpus_dir)
        gold_path = Path(payload.gold_path)
        if not corpus_dir.is_dir() or not gold_path.is_file():
            raise HTTPException(status_code=404, detail="corpus_dir or gold_path not found")
        return run_eval(corpus_dir, gold_path, k=payload.k, base_config=eng.config)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    # -----------------------------------------------------------------------
    # FARLOOM extensions
    # -----------------------------------------------------------------------

    @app.get("/docs/list", response_model=list[Document])
    def list_documents(eng: EngineDep) -> list[Document]:
        """List all ingested documents with chunk counts."""
        return eng.store.list_documents()

    @app.get("/graph/stats", response_model=GraphStats)
    def graph_stats(eng: EngineDep) -> GraphStats:
        """Return entity/edge counts and entity labels."""
        return GraphStats(
            num_nodes=eng.graph.num_nodes,
            num_edges=eng.graph.num_edges,
            entities=eng.graph.entities()[:200],  # cap for large corpora
        )

    @app.get("/graph/entities")
    def graph_entities(
        eng: EngineDep,
        q: Annotated[str, Query(description="Search filter (case-insensitive)")] = "",
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
    ) -> dict[str, list[str]]:
        """Search entity labels, returning up to `limit` matches."""
        entities = eng.graph.entities()
        if q:
            q_lower = q.lower()
            entities = [e for e in entities if q_lower in e.lower()]
        return {"entities": entities[:limit]}

    @app.get("/status", response_model=ProviderStatus)
    def provider_status(eng: EngineDep) -> ProviderStatus:
        """Return current provider and demo-mode status."""
        is_demo = eng.config.provider == "fake"
        return ProviderStatus(
            provider=eng.config.provider,
            demo_mode=is_demo,
            version=__version__,
            data_dir=str(eng.config.data_dir),
        )

    @app.get("/chunks/count")
    def chunk_count(eng: EngineDep) -> dict[str, int]:
        """Total indexed chunk count."""
        return {"count": eng.vectors.count()}

    # -----------------------------------------------------------------------
    # Frontend: serve static files and SPA index
    # -----------------------------------------------------------------------

    _static_dir = Path(__file__).parent / "static"
    if _static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        def frontend_root() -> FileResponse:
            return FileResponse(str(_static_dir / "index.html"))

    return app


app = create_app()
