"""FastAPI surface tests via TestClient (no server, no network)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from graphrag_knowledge_engine.api import create_app
from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.engine import create_engine
from tests.conftest import CORPUS_DIR


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    engine = create_engine(
        EngineConfig(data_dir=tmp_path_factory.mktemp("api-engine"), provider="fake")
    )
    app = create_app(engine=engine)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_ingest_then_ask(client: TestClient) -> None:
    for doc in sorted(CORPUS_DIR.glob("*.md")):
        resp = client.post("/ingest", json={"path": str(doc), "strategy": "heading"})
        assert resp.status_code == 200
        assert resp.json()["num_chunks"] > 0

    resp = client.get("/ask", params={"q": "Who owns the Meridian Policy?"})
    assert resp.status_code == 200
    body = resp.json()
    assert not body["refused"]
    assert "Priya Raghavan" in body["answer"]
    assert body["citations"]
    assert body["debug"]["mode"] == "hybrid_graph"


def test_ask_validates_mode(client: TestClient) -> None:
    resp = client.get("/ask", params={"q": "hello", "mode": "nonsense"})
    assert resp.status_code == 422


def test_ingest_missing_file_404(client: TestClient) -> None:
    resp = client.post("/ingest", json={"path": "/nonexistent/file.md"})
    assert resp.status_code == 404


def test_ingest_rejects_unknown_extension(client: TestClient, tmp_path: Path) -> None:
    bad = tmp_path / "doc.docx"
    bad.write_text("hello")
    resp = client.post("/ingest", json={"path": str(bad)})
    assert resp.status_code == 422


def test_ingest_rejects_extra_fields(client: TestClient) -> None:
    resp = client.post("/ingest", json={"path": "x.md", "surprise": True})
    assert resp.status_code == 422


def test_graph_neighbors(client: TestClient) -> None:
    resp = client.get("/graph/neighbors", params={"entity": "Helios Sync", "depth": 1})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"]
    assert body["edges"]


def test_eval_run_endpoint_validates_paths(client: TestClient) -> None:
    resp = client.post(
        "/eval/run", json={"gold_path": "missing.json", "corpus_dir": "missing", "k": 5}
    )
    assert resp.status_code == 404


def test_eval_run_endpoint_scores_all_modes(client: TestClient) -> None:
    from tests.conftest import GOLD_PATH

    resp = client.post(
        "/eval/run",
        json={"gold_path": str(GOLD_PATH), "corpus_dir": str(CORPUS_DIR), "k": 5},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [r["mode"] for r in body["results"]] == ["dense", "bm25", "hybrid", "hybrid_graph"]
    assert all(0.0 <= r["hit_at_k"] <= 1.0 for r in body["results"])


def test_delete_document_endpoint(client: TestClient) -> None:
    # 1. Fetch documents
    docs_resp = client.get("/docs/list")
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert len(docs) > 0
    target_doc = docs[0]
    doc_id = target_doc["doc_id"]

    # 2. Delete document
    del_resp = client.delete(f"/docs/{doc_id}")
    assert del_resp.status_code == 200
    assert del_resp.json() == {"success": True, "deleted": doc_id}

    # 3. Verify removed from list
    docs_resp2 = client.get("/docs/list")
    assert not any(d["doc_id"] == doc_id for d in docs_resp2.json())

    # 4. Deleting non-existent returns 404
    del_missing = client.delete("/docs/non_existent_doc_id_123")
    assert del_missing.status_code == 404

