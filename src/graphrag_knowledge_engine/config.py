"""Engine configuration.

Every knob lives here so the API, CLI, tests, and eval harness construct the
engine the same way. Values come from explicit arguments first, environment
variables second, sane defaults last.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

ProviderName = Literal["fake", "openai", "gemini"]
GraphBackend = Literal["networkx", "neo4j"]


class EngineConfig(BaseModel):
    """Configuration for a single engine instance."""

    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Field(default=Path("artifacts/runtime"))
    provider: ProviderName = "fake"

    # Graph Backend
    graph_backend: GraphBackend = "networkx"
    neo4j_uri: str = ""
    neo4j_username: str = ""
    neo4j_password: str = ""
    neo4j_database: str = "neo4j"
    neo4j_connection_timeout: float = 10.0

    # Embeddings
    embed_dim: int = 512  # FakeEmbedder dimension; real providers report their own.

    # Chunking defaults
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    dense_top_k: int = 10
    bm25_top_k: int = 10
    graph_top_k: int = 6
    rrf_k: int = 60
    dense_sim_floor: float = 0.05  # drop dense hits with cosine similarity below this
    graph_depth: int = 2
    graph_max_nodes: int = 25

    # Answering
    context_size: int = 5
    # Refuse when normalized retrieval confidence (max fused score over the
    # mode's all-legs-rank-1 ceiling, in [0, 1]) falls below this.
    min_confidence: float = 0.4

    # OpenAI (real providers only)
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Gemini (free-tier provider)
    gemini_api_key: str = ""
    gemini_chat_model: str = "gemini-2.0-flash-lite"


def config_from_env(data_dir: Path | None = None) -> EngineConfig:
    """Build a config from environment variables (see .env.example)."""
    load_dotenv()
    raw_provider = os.environ.get("GRAPHRAG_PROVIDER", "fake").lower()
    if raw_provider == "openai":
        provider: ProviderName = "openai"
    elif raw_provider == "gemini":
        provider = "gemini"
    else:
        provider = "fake"

    raw_backend = os.environ.get("GRAPH_BACKEND", "networkx").strip().lower()
    graph_backend: GraphBackend = "neo4j" if raw_backend == "neo4j" else "networkx"

    return EngineConfig(
        data_dir=data_dir or Path(os.environ.get("GRAPHRAG_DATA_DIR", "artifacts/runtime")),
        provider=provider,
        graph_backend=graph_backend,
        neo4j_uri=(os.environ.get("NEO4J_URI") or os.environ.get("NEO4J_URL") or "").strip(),
        neo4j_username=(os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER") or "").strip(),
        neo4j_password=os.environ.get("NEO4J_PASSWORD", "").strip(),
        neo4j_database=(os.environ.get("NEO4J_DATABASE") or "neo4j").strip(),
        neo4j_connection_timeout=float(os.environ.get("NEO4J_CONNECTION_TIMEOUT", "10.0")),
        min_confidence=float(os.environ.get("GRAPHRAG_MIN_CONFIDENCE", "0.4")),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_embed_model=os.environ.get("GRAPHRAG_OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        openai_chat_model=os.environ.get("GRAPHRAG_OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        gemini_chat_model=os.environ.get("GRAPHRAG_GEMINI_MODEL", "gemini-2.0-flash-lite"),
    )

