"""Tests for Neo4j knowledge graph store, driver lifecycle, and Cypher queries.

Includes unit tests, mock driver integration tests, and live AuraDB tests (which
skip cleanly if live credentials are not present in the environment).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest

from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.models import NeighborEdge, Triple
from graphrag_knowledge_engine.store.factory import create_graph_store
from graphrag_knowledge_engine.store.neo4j_store import (
    Neo4jGraphStore,
    close_all_drivers,
    get_shared_driver,
    redact_uri,
)


def test_redact_uri() -> None:
    uri_with_pass = "neo4j+s://neo4j:SuperSecret123!@ab12cd34.databases.neo4j.io:7687"
    redacted = redact_uri(uri_with_pass)
    assert "SuperSecret123!" not in redacted
    assert "neo4j:****@ab12cd34.databases.neo4j.io" in redacted

    clean_uri = "bolt://localhost:7687"
    assert redact_uri(clean_uri) == "bolt://localhost:7687"


def test_driver_cache_and_close() -> None:
    # Verify driver cache registry
    close_all_drivers()
    driver1 = get_shared_driver("bolt://localhost:7687", "neo4j", "pass")
    driver2 = get_shared_driver("bolt://localhost:7687", "neo4j", "pass")
    assert driver1 is driver2
    close_all_drivers()


def test_factory_missing_credentials_raises() -> None:
    cfg = EngineConfig(graph_backend="neo4j", neo4j_uri="", neo4j_password="")
    with pytest.raises(RuntimeError, match="NEO4J_URI is not set"):
        create_graph_store(cfg)

    cfg2 = EngineConfig(
        graph_backend="neo4j",
        neo4j_uri="neo4j+s://test.databases.neo4j.io",
        neo4j_password="",
    )
    with pytest.raises(RuntimeError, match="NEO4J_PASSWORD is empty"):
        create_graph_store(cfg2)


def test_factory_defaults_to_networkx() -> None:
    cfg = EngineConfig(graph_backend="networkx")
    store = create_graph_store(cfg)
    assert store.backend_name == "networkx"
    assert store.ping() is True


# -- Mock driver test fixtures -----------------------------------------------


class MockRecord(dict[str, Any]):
    def __getitem__(self, key: str) -> Any:
        return super().__getitem__(key)


class MockResult:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self._records = [MockRecord(r) for r in (records or [])]
        self._consumed = False

    def __iter__(self) -> Any:
        return iter(self._records)

    def single(self) -> MockRecord | None:
        return self._records[0] if self._records else None

    def consume(self) -> None:
        self._consumed = True


@pytest.fixture
def mock_driver() -> MagicMock:
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    return driver


def test_neo4j_store_init_schema(mock_driver: MagicMock) -> None:
    session = mock_driver.session.return_value.__enter__.return_value
    session.run.return_value = MockResult([])

    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        driver=mock_driver,
    )
    store.init_schema()

    assert session.run.call_count == 3
    executed_queries = [call[0][0] for call in session.run.call_args_list]
    assert any("CONSTRAINT" in q for q in executed_queries)
    assert any("INDEX" in q for q in executed_queries)


def test_neo4j_store_ping(mock_driver: MagicMock) -> None:
    session = mock_driver.session.return_value.__enter__.return_value
    session.run.return_value = MockResult([{"ping": 1}])

    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        driver=mock_driver,
    )
    assert store.ping() is True

    # When query fails
    session.run.side_effect = Exception("Connection lost")
    assert store.ping() is False


def test_neo4j_store_verify_connectivity_diagnoses_auradb(mock_driver: MagicMock) -> None:
    from neo4j.exceptions import ServiceUnavailable

    mock_driver.verify_connectivity.side_effect = ServiceUnavailable("Cannot reach server")

    store = Neo4jGraphStore(
        uri="neo4j+s://xyz123.databases.neo4j.io",
        username="neo4j",
        password="password",
        driver=mock_driver,
    )
    with pytest.raises(ConnectionError, match="AuraDB instance, it may have been paused"):
        store.verify_connectivity()


def test_neo4j_store_add_triples_and_mentions(mock_driver: MagicMock) -> None:
    session = mock_driver.session.return_value.__enter__.return_value
    session.run.return_value = MockResult([])

    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        space_id="test-space",
        driver=mock_driver,
    )

    triples = [
        Triple(
            subject="Helios Sync",
            relation="replaces",
            obj="Legacy Beacon",
            chunk_id="chunk_001",
            doc_id="doc_1",
        )
    ]
    store.add_triples(triples)
    assert session.run.called
    kwargs = session.run.call_args[1]
    assert kwargs["space_id"] == "test-space"
    batch = kwargs["batch"]
    assert len(batch) == 1
    assert batch[0]["s_key"] == "helios sync"
    assert batch[0]["o_key"] == "legacy beacon"
    assert batch[0]["relation"] == "replaces"
    assert batch[0]["chunk_ids"] == ["chunk_001"]

    # Mentions
    store.add_mentions([("Helios Sync", "chunk_002")])
    assert session.run.called
    m_kwargs = session.run.call_args[1]
    m_batch = m_kwargs["batch"]
    assert m_batch[0]["key"] == "helios sync"
    assert m_batch[0]["chunk_ids"] == ["chunk_002"]


def test_neo4j_store_counts_and_entities(mock_driver: MagicMock) -> None:
    session = mock_driver.session.return_value.__enter__.return_value
    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        driver=mock_driver,
    )

    # num_nodes
    session.run.return_value = MockResult([{"cnt": 42}])
    assert store.num_nodes == 42

    # num_edges
    session.run.return_value = MockResult([{"cnt": 18}])
    assert store.num_edges == 18

    # entities
    session.run.return_value = MockResult([{"label": "Alpha"}, {"label": "Beta"}])
    assert store.entities() == ["Alpha", "Beta"]

    # has_entity
    session.run.return_value = MockResult([{"exists": True}])
    assert store.has_entity("Alpha") is True


def test_neo4j_store_detect_query_entities(mock_driver: MagicMock) -> None:
    session = mock_driver.session.return_value.__enter__.return_value
    session.run.return_value = MockResult(
        [
            {"key": "helios", "label": "Helios"},
            {"key": "helios sync", "label": "Helios Sync"},
            {"key": "beacon", "label": "Beacon"},
        ]
    )

    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        driver=mock_driver,
    )

    # Longest match wins: "Helios Sync" matches before "Helios"
    found = store.detect_query_entities("How does Helios Sync interact with Beacon?")
    assert "Helios Sync" in found
    assert "Beacon" in found
    assert "Helios" not in found


def test_neo4j_store_expand(mock_driver: MagicMock) -> None:
    session = mock_driver.session.return_value.__enter__.return_value

    # Traversal query returns records with hops and chunks
    traversal_records = [
        {"key": "seed1", "label": "Seed 1", "chunks": ["c1", "c2"], "hop": 0},
        {"key": "neighbor1", "label": "Neighbor 1", "chunks": ["c2", "c3"], "hop": 1},
    ]
    # Focus query returns degree of connected nodes
    focus_records = [
        {"key": "seed1", "chunks": ["c1", "c2"], "degree": 2},
        {"key": "neighbor1", "chunks": ["c2", "c3"], "degree": 2},
    ]

    session.run.side_effect = [
        MockResult(traversal_records),
        MockResult(focus_records),
    ]

    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        driver=mock_driver,
    )

    results = store.expand(["Seed 1"], max_depth=2, max_nodes=25, top_k=5)
    assert len(results) > 0
    # Seed chunks have highest closeness + focus
    chunk_ids = [cid for cid, _ in results]
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids


def test_neo4j_store_neighbors(mock_driver: MagicMock) -> None:
    session = mock_driver.session.return_value.__enter__.return_value
    session.run.return_value = MockResult(
        [
            {
                "subject": "Alpha",
                "relation": "depends_on",
                "obj": "Beta",
                "chunk_ids": ["c1", "c2"],
            }
        ]
    )

    store = Neo4jGraphStore(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password",
        driver=mock_driver,
    )

    edges = store.neighbors("Alpha", depth=1)
    assert len(edges) == 1
    assert isinstance(edges[0], NeighborEdge)
    assert edges[0].subject == "Alpha"
    assert edges[0].relation == "depends_on"
    assert edges[0].obj == "Beta"
    assert edges[0].chunk_ids == ["c1", "c2"]


def test_neo4j_store_diagnostics(mock_driver: MagicMock) -> None:
    session = mock_driver.session.return_value.__enter__.return_value
    session.run.side_effect = [
        MockResult([{"ping": 1}]),  # ping
        MockResult([{"cnt": 10}]),  # num_nodes
        MockResult([{"cnt": 5}]),  # num_edges
    ]

    store = Neo4jGraphStore(
        uri="neo4j+s://neo4j:secret@test.databases.neo4j.io:7687",
        username="neo4j",
        password="secret",
        database="neo4j",
        space_id="default",
        driver=mock_driver,
    )

    diag = store.get_diagnostics()
    assert diag["backend"] == "neo4j"
    assert diag["status"] == "connected"
    assert diag["ready"] is True
    assert "secret" not in diag["uri"]
    assert diag["num_nodes"] == 10
    assert diag["num_edges"] == 5


def test_api_endpoints_under_neo4j_mode(mock_driver: MagicMock, tmp_path: Any) -> None:
    from starlette.testclient import TestClient

    from graphrag_knowledge_engine.api import create_app
    from graphrag_knowledge_engine.engine import Engine
    from graphrag_knowledge_engine.providers.embeddings import FakeEmbedder
    from graphrag_knowledge_engine.providers.llm import FakeLLM

    session = mock_driver.session.return_value.__enter__.return_value
    session.run.return_value.single.return_value = {"ping": 1, "cnt": 100, "exists": True}
    session.run.return_value.__iter__.return_value = [
        {"label": "Helios"},
        {"subject": "A", "relation": "r", "obj": "B", "chunk_ids": ["c1"]},
    ]

    cfg = EngineConfig(data_dir=tmp_path, graph_backend="networkx")
    eng = Engine(cfg, FakeEmbedder(512), FakeLLM())
    eng.config = eng.config.model_copy(update={"graph_backend": "neo4j"})
    eng.graph = Neo4jGraphStore(
        uri="neo4j+s://test.databases.neo4j.io",
        username="neo4j",
        password="secret",
        driver=mock_driver,
    )

    app = create_app(engine=eng)
    with TestClient(app) as client:
        status = client.get("/status").json()
        assert status["graph_backend"] == "neo4j"
        assert status["neo4j_status"] == "connected"
        assert status["graph_diagnostics"]["ready"] is True

        ready_res = client.get("/health/ready")
        assert ready_res.status_code == 200
        assert ready_res.json()["status"] == "ready"
        assert ready_res.json()["graph_backend"] == "neo4j"

        health_res = client.get("/health")
        assert health_res.status_code == 200
        assert health_res.json()["graph_backend"] == "neo4j"
        assert health_res.json()["graph_ready"] is True



# -- Live Neo4j AuraDB integration test (skips if no credentials) -----------

@pytest.mark.skipif(
    not os.environ.get("NEO4J_URI"),
    reason="Live Neo4j test requires NEO4J_URI environment variable",
)
def test_live_neo4j_auradb_integration() -> None:
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USERNAME") or os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    database = os.environ.get("NEO4J_DATABASE", "neo4j")

    store = Neo4jGraphStore(
        uri=uri,
        username=user,
        password=password,
        database=database,
        space_id="test_live_verify",
    )
    store.verify_connectivity()
    store.init_schema()

    # Test write
    triple = Triple(
        subject="FARLOOM Verification",
        relation="validates",
        obj="Neo4j AuraDB Engine",
        chunk_id="chunk_verify_01",
        doc_id="doc_verify",
    )
    store.add_triples([triple])

    # Test read
    assert store.has_entity("FARLOOM Verification") is True
    neighbors = store.neighbors("FARLOOM Verification", depth=1)
    assert len(neighbors) >= 1
    assert neighbors[0].relation == "validates"

    # Test expansion
    expanded = store.expand(["FARLOOM Verification"], max_depth=2, top_k=5)
    assert len(expanded) >= 1
    assert expanded[0][0] == "chunk_verify_01"

    store.close()
