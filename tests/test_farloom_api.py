"""Tests for FARLOOM-specific API extensions."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from graphrag_knowledge_engine.api import create_app
from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.engine import create_engine
from tests.conftest import CORPUS_DIR


@pytest.fixture(scope="module")
def farloom_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    engine = create_engine(
        EngineConfig(data_dir=tmp_path_factory.mktemp("farloom-api"), provider="fake")
    )
    # Ingest corpus so graph/docs endpoints return real data
    for doc in sorted(CORPUS_DIR.glob("*.md")):
        engine.ingest_path(doc, strategy="heading")
    app = create_app(engine=engine)
    with TestClient(app) as client:
        yield client


def test_farloom_health(farloom_client: TestClient) -> None:
    resp = farloom_client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_docs_list(farloom_client: TestClient) -> None:
    resp = farloom_client.get("/docs/list")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    assert len(docs) >= 3  # bundled corpus has 3+ docs
    assert all("source" in d and "num_chunks" in d for d in docs)


def test_graph_stats(farloom_client: TestClient) -> None:
    resp = farloom_client.get("/graph/stats")
    assert resp.status_code == 200
    body = resp.json()
    assert body["num_nodes"] > 0
    assert body["num_edges"] >= 0
    assert isinstance(body["entities"], list)


def test_graph_entities_no_filter(farloom_client: TestClient) -> None:
    resp = farloom_client.get("/graph/entities")
    assert resp.status_code == 200
    body = resp.json()
    assert "entities" in body
    assert len(body["entities"]) > 0


def test_graph_entities_with_filter(farloom_client: TestClient) -> None:
    resp = farloom_client.get("/graph/entities", params={"q": "aurora", "limit": 10})
    assert resp.status_code == 200
    body = resp.json()
    assert "entities" in body
    # All returned entities should match filter (case-insensitive)
    for e in body["entities"]:
        assert "aurora" in e.lower()


def test_provider_status(farloom_client: TestClient) -> None:
    resp = farloom_client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "fake"
    assert body["demo_mode"] is True
    assert "version" in body


def test_chunk_count(farloom_client: TestClient) -> None:
    resp = farloom_client.get("/chunks/count")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0


def test_frontend_serves_index(farloom_client: TestClient) -> None:
    """The root URL should serve the FARLOOM frontend HTML."""
    resp = farloom_client.get("/")
    assert resp.status_code == 200
    assert b"FARLOOM" in resp.content


def test_graph_entities_limit_validated(farloom_client: TestClient) -> None:
    resp = farloom_client.get("/graph/entities", params={"limit": 600})
    assert resp.status_code == 422  # limit > 500


def test_ask_all_modes_farloom(farloom_client: TestClient) -> None:
    """Each retrieval mode returns a valid response (not necessarily answered)."""
    for mode in ["dense", "bm25", "hybrid", "hybrid_graph"]:
        resp = farloom_client.get(
            "/ask",
            params={"q": "What is Helios Sync?", "mode": mode},
        )
        assert resp.status_code == 200, f"mode={mode} failed"
        body = resp.json()
        assert "answer" in body
        assert "citations" in body
        assert "debug" in body
