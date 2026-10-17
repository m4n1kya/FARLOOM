"""FastAPI surface: /ingest, /ask, /graph/neighbors, /eval/run, /docs/list,
/graph/stats, /graph/entities, /spaces, /upload, plus static frontend serving.

The engine is created once per process from environment config and shared via
``app.state``. Supports isolated multi-tenant Knowledge Spaces.
``create_app`` accepts an engine override so tests can inject
an engine built on fakes in a temp directory.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from graphrag_knowledge_engine import __version__
from graphrag_knowledge_engine.config import config_from_env
from graphrag_knowledge_engine.engine import Engine
from graphrag_knowledge_engine.evals.runner import run_eval
from graphrag_knowledge_engine.models import (
    AskResponse,
    ChunkStrategy,
    Document,
    EvalReport,
    EvalRunRequest,
    IngestRequest,
    IngestResponse,
    NeighborsResponse,
    RetrievalMode,
)
from graphrag_knowledge_engine.spaces import SpaceManager, SpaceMetadata


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


class CreateSpaceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=60)
    description: str = Field(default="", max_length=250)


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


def get_engine(
    request: Request,
    space: Annotated[str | None, Query(description="Target Knowledge Space")] = None,
) -> Engine:
    """Resolve the Engine for the requested Knowledge Space."""
    target_space = space or request.headers.get("x-farloom-space") or "default"
    spaces_mgr: SpaceManager | None = getattr(request.app.state, "spaces", None)
    if spaces_mgr is not None:
        try:
            return spaces_mgr.get_engine(target_space)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Knowledge Space '{target_space}' not found") from exc

    eng: Engine = request.app.state.engine
    return eng


EngineDep = Annotated[Engine, Depends(get_engine)]


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(engine: Engine | None = None) -> FastAPI:
    base_config = config_from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        spaces_mgr: SpaceManager
        if engine is not None:
            # Explicit engine injected (e.g. test suite)
            spaces_mgr = SpaceManager(engine.config.data_dir, engine.config)
            spaces_mgr.set_engine("default", engine)
            app.state.engine = engine
            app.state.spaces = spaces_mgr
        else:
            spaces_mgr = SpaceManager(base_config.data_dir, base_config)
            default_engine = spaces_mgr.get_engine("default")
            _auto_ingest_corpus(default_engine)
            app.state.engine = default_engine
            app.state.spaces = spaces_mgr

        yield

        spaces_mgr.close()

    app = FastAPI(
        title="FARLOOM — Technical Knowledge Intelligence Engine",
        version=__version__,
        description="Hybrid RAG: dense + BM25 + knowledge graph, fused with RRF.",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Knowledge Spaces management endpoints
    # -----------------------------------------------------------------------

    @app.get("/spaces", response_model=list[SpaceMetadata])
    def list_spaces(request: Request) -> list[SpaceMetadata]:
        """List all isolated Knowledge Spaces with document and graph counts."""
        spaces_mgr: SpaceManager = request.app.state.spaces
        return spaces_mgr.list_spaces()

    @app.post("/spaces", response_model=SpaceMetadata)
    def create_space(payload: CreateSpaceRequest, request: Request) -> SpaceMetadata:
        """Create a new isolated Knowledge Space."""
        spaces_mgr: SpaceManager = request.app.state.spaces
        return spaces_mgr.create_space(payload.name, payload.description)

    @app.delete("/spaces/{space_id}")
    def delete_space(space_id: str, request: Request) -> dict[str, Any]:
        """Delete an isolated Knowledge Space and its indexed data."""
        spaces_mgr: SpaceManager = request.app.state.spaces
        try:
            success = spaces_mgr.delete_space(space_id)
            if not success:
                raise HTTPException(status_code=404, detail=f"Space '{space_id}' not found")
            return {"success": True, "deleted": space_id}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -----------------------------------------------------------------------
    # Document Upload endpoint
    # -----------------------------------------------------------------------

    @app.post("/spaces/{space_id}/upload", response_model=IngestResponse)
    @app.post("/upload", response_model=IngestResponse)
    async def upload_document(
        request: Request,
        file: Annotated[UploadFile, File(description="Document to ingest (.pdf, .md, .txt)")],
        strategy: Annotated[ChunkStrategy, Form()] = "heading",
        space_id: str | None = None,
    ) -> IngestResponse:
        """Upload and ingest a document into an isolated Knowledge Space."""
        target_space = space_id or request.query_params.get("space") or "default"
        spaces_mgr: SpaceManager = request.app.state.spaces
        try:
            space_dir = spaces_mgr.get_space_dir(target_space)
            eng = spaces_mgr.get_engine(target_space)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"Space '{target_space}' not found") from exc

        # Validate file extension
        filename = file.filename or "uploaded_doc.txt"
        suffix = Path(filename).suffix.lower()
        if suffix not in (".pdf", ".md", ".txt"):
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{suffix}'. Allowed: .pdf, .md, .txt",
            )

        # Save to space's uploads folder
        upload_dest = space_dir / "uploads" / filename
        with upload_dest.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        try:
            return eng.ingest_path(upload_dest, strategy=strategy)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to ingest document: {exc}") from exc

    # -----------------------------------------------------------------------
    # Retrieval and Question Answering endpoints
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
    # Space-aware Information & Diagnostics endpoints
    # -----------------------------------------------------------------------

    @app.get("/docs/list", response_model=list[Document])
    def list_documents(eng: EngineDep) -> list[Document]:
        """List all ingested documents in the target space."""
        return eng.store.list_documents()

    @app.get("/graph/stats", response_model=GraphStats)
    def graph_stats(eng: EngineDep) -> GraphStats:
        """Return entity/edge counts and entity labels in the target space."""
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
        """Search entity labels in the target space, returning up to `limit` matches."""
        entities = eng.graph.entities()
        if q:
            q_lower = q.lower()
            entities = [e for e in entities if q_lower in e.lower()]
        return {"entities": entities[:limit]}

    @app.get("/status", response_model=ProviderStatus)
    def provider_status(eng: EngineDep) -> ProviderStatus:
        """Return current provider, space directory, and demo-mode status."""
        is_demo = eng.config.provider == "fake"
        return ProviderStatus(
            provider=eng.config.provider,
            demo_mode=is_demo,
            version=__version__,
            data_dir=str(eng.config.data_dir),
        )

    @app.get("/chunks/count")
    def chunk_count(eng: EngineDep) -> dict[str, int]:
        """Total indexed chunk count in the target space."""
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
