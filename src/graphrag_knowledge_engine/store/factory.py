"""Factory for constructing the active graph store backend."""

from __future__ import annotations

import logging

from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.models import Triple
from graphrag_knowledge_engine.store.base import BaseGraphStore
from graphrag_knowledge_engine.store.graph import NetworkXGraphStore
from graphrag_knowledge_engine.store.neo4j_store import Neo4jGraphStore

logger = logging.getLogger(__name__)


def create_graph_store(
    config: EngineConfig,
    space_id: str = "default",
    triples: list[Triple] | None = None,
    mentions: list[tuple[str, str]] | None = None,
) -> BaseGraphStore:
    """Instantiate the configured graph store backend.

    If GRAPH_BACKEND=neo4j:
    - Verifies credentials are provided.
    - Connects to Neo4j / AuraDB.
    - Verifies connectivity.
    - Initializes idempotent schema constraints and indexes.
    - Syncs existing SQLite triples and mentions if provided.

    If GRAPH_BACKEND=networkx (default):
    - Initializes in-memory NetworkX store from SQLite rows.
    """
    backend = config.graph_backend.lower().strip()

    if backend == "neo4j":
        if not config.neo4j_uri:
            raise RuntimeError(
                "GRAPH_BACKEND is set to 'neo4j', but NEO4J_URI is not set. "
                "Please configure NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD, "
                "or set GRAPH_BACKEND=networkx for local offline execution."
            )
        if not config.neo4j_password:
            raise RuntimeError(
                "GRAPH_BACKEND is set to 'neo4j', but NEO4J_PASSWORD is empty. "
                "Please configure NEO4J_PASSWORD or set GRAPH_BACKEND=networkx."
            )

        store = Neo4jGraphStore(
            uri=config.neo4j_uri,
            username=config.neo4j_username,
            password=config.neo4j_password,
            database=config.neo4j_database,
            space_id=space_id,
            connection_timeout=config.neo4j_connection_timeout,
        )
        # Explicit connectivity check on startup
        store.verify_connectivity()
        # Initialize schema
        store.init_schema()

        if triples:
            store.add_triples(triples)
        if mentions:
            store.add_mentions(mentions)

        return store

    # Default: NetworkX
    return NetworkXGraphStore.from_store_rows(triples or [], mentions or [])
