"""Tests for isolated multi-tenant Knowledge Spaces and document uploads."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graphrag_knowledge_engine.api import create_app
from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.engine import create_engine


@pytest.fixture
def api_client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    base_dir = tmp_path_factory.mktemp("farloom-spaces")
    config = EngineConfig(data_dir=base_dir, provider="fake")
    engine = create_engine(config)
    app = create_app(engine)
    with TestClient(app) as client:
        yield client


def test_list_spaces_initially(api_client: TestClient) -> None:
    resp = api_client.get("/spaces")
    assert resp.status_code == 200
    spaces = resp.json()
    assert isinstance(spaces, list)
    assert any(s["id"] == "default" for s in spaces)


def test_create_and_delete_space(api_client: TestClient) -> None:
    # Create
    create_resp = api_client.post(
        "/spaces",
        json={"name": "Project Alpha", "description": "Alpha team documentation"},
    )
    assert create_resp.status_code == 200
    space = create_resp.json()
    assert space["id"] == "project-alpha"
    assert space["name"] == "Project Alpha"

    # Verify in list
    list_resp = api_client.get("/spaces")
    assert any(s["id"] == "project-alpha" for s in list_resp.json())

    # Cannot delete default
    del_default = api_client.delete("/spaces/default")
    assert del_default.status_code == 400

    # Delete custom space
    del_resp = api_client.delete("/spaces/project-alpha")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] == "project-alpha"

    # Verify gone
    list_after = api_client.get("/spaces")
    assert not any(s["id"] == "project-alpha" for s in list_after.json())


def test_spaces_isolation(api_client: TestClient, tmp_path: Path) -> None:
    # Create two spaces
    api_client.post("/spaces", json={"name": "Space Alpha"})
    api_client.post("/spaces", json={"name": "Space Beta"})

    # Upload Doc A into Space Alpha
    doc_a_content = b"# Alpha Architecture\n\nAlphaNet connects Nova Cluster to Pulse Gateway."
    resp_a = api_client.post(
        "/spaces/space-alpha/upload",
        files={"file": ("alpha_doc.md", io.BytesIO(doc_a_content), "text/markdown")},
        data={"strategy": "heading"},
    )
    assert resp_a.status_code == 200
    assert resp_a.json()["num_chunks"] > 0

    # Upload Doc B into Space Beta
    doc_b_content = b"# Beta Specification\n\nBetaProtocol governs Quantum Relay nodes."
    resp_b = api_client.post(
        "/spaces/space-beta/upload",
        files={"file": ("beta_doc.md", io.BytesIO(doc_b_content), "text/markdown")},
        data={"strategy": "heading"},
    )
    assert resp_b.status_code == 200
    assert resp_b.json()["num_chunks"] > 0

    # Verify Space Alpha docs
    docs_a = api_client.get("/docs/list", params={"space": "space-alpha"}).json()
    assert len(docs_a) == 1
    assert docs_a[0]["source"] == "alpha_doc.md"

    # Verify Space Beta docs
    docs_b = api_client.get("/docs/list", params={"space": "space-beta"}).json()
    assert len(docs_b) == 1
    assert docs_b[0]["source"] == "beta_doc.md"

    # Verify Space Alpha does NOT have Quantum Relay in graph entities
    entities_a = api_client.get("/graph/entities", params={"space": "space-alpha"}).json()["entities"]
    entities_b = api_client.get("/graph/entities", params={"space": "space-beta"}).json()["entities"]
    assert "Nova Cluster" in entities_a or "AlphaNet" in entities_a
    assert "Quantum Relay" in entities_b or "BetaProtocol" in entities_b
    assert "Quantum Relay" not in entities_a

    # Querying Space Alpha for AlphaNet succeeds
    ask_a = api_client.get("/ask", params={"q": "What connects Nova Cluster?", "space": "space-alpha"}).json()
    assert "Nova Cluster" in ask_a["answer"] or len(ask_a["citations"]) > 0


def test_upload_unsupported_file_type(api_client: TestClient) -> None:
    resp = api_client.post(
        "/spaces/default/upload",
        files={"file": ("malicious.exe", io.BytesIO(b"MZ..."), "application/octet-stream")},
    )
    assert resp.status_code == 422
    assert "Unsupported file type" in resp.json()["detail"]


def test_upload_with_auto_space_creation(api_client: TestClient) -> None:
    doc_content = b"# Gamma Manual\n\nGamma Core handles telemetry routing."
    resp = api_client.post(
        "/upload",
        files={"file": ("gamma.md", io.BytesIO(doc_content), "text/markdown")},
        data={
            "strategy": "heading",
            "create_new_space": "true",
            "new_space_name": "Project Gamma",
            "new_space_preset": "tech_docs",
        },
    )
    assert resp.status_code == 200
    assert resp.headers.get("x-farloom-space-id") == "project-gamma"

    # Verify project-gamma exists and has gamma.md
    docs = api_client.get("/docs/list", params={"space": "project-gamma"}).json()
    assert len(docs) == 1
    assert docs[0]["source"] == "gamma.md"

    # Verify default space does NOT have gamma.md
    default_docs = api_client.get("/docs/list", params={"space": "default"}).json()
    assert not any(d["source"] == "gamma.md" for d in default_docs)

