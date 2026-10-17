"""End-to-end engine behaviour over the bundled corpus (offline fakes)."""

from graphrag_knowledge_engine.engine import Engine


def test_ingest_populates_all_three_indexes(corpus_engine: Engine) -> None:
    assert len(corpus_engine.store.all_chunks()) > 10
    assert corpus_engine.vectors.count() == len(corpus_engine.store.all_chunks())
    assert corpus_engine.graph.num_nodes > 10
    assert corpus_engine.graph.num_edges > 10


def test_single_hop_answer_with_citations(corpus_engine: Engine) -> None:
    response = corpus_engine.ask("What rate limit does Beacon API enforce per customer agent?")
    assert not response.refused
    assert "1200" in response.answer
    assert response.citations, "expected at least one citation"
    tags = {c.tag for c in response.citations}
    assert any(f"[{t}]" in response.answer for t in tags)
    assert all(c.source.endswith(".md") for c in response.citations)


def test_low_confidence_refusal(corpus_engine: Engine) -> None:
    response = corpus_engine.ask("What is the boiling point of mercury on Venus?")
    assert response.refused
    assert response.citations == []
    assert response.debug.confidence < corpus_engine.config.min_confidence


def test_refusal_on_empty_corpus(empty_engine: Engine) -> None:
    response = empty_engine.ask("Who is the CEO?")
    assert response.refused
    assert response.debug.fused == []


def test_multi_hop_question_needs_the_graph(corpus_engine: Engine) -> None:
    """The HX-9 evidence chunk never mentions Daniel Okafor: the graph path
    (Daniel Okafor -> Cascade Campus -> HX-9 deployment chunk) ranks it up."""
    question = (
        "What computing gear and storage fill the racks "
        "at the location where Daniel Okafor is based?"
    )

    def evidence_rank(mode: str) -> int | None:
        resp = corpus_engine.ask(question, mode=mode)  # type: ignore[arg-type]
        chunks = corpus_engine.store.get_chunks([c.chunk_id for c in resp.debug.fused])
        for i, cand in enumerate(resp.debug.fused):
            chunk = chunks.get(cand.chunk_id)
            if chunk is not None and "HX-9" in chunk.text:
                return i + 1
        return None

    hybrid_rank = evidence_rank("hybrid")
    graph_rank = evidence_rank("hybrid_graph")
    assert graph_rank is not None and graph_rank <= 5
    assert hybrid_rank is None or graph_rank < hybrid_rank


def test_graph_entities_detected_in_query(corpus_engine: Engine) -> None:
    response = corpus_engine.ask("Which subsystems make up Helios Sync?")
    assert "Helios Sync" in response.debug.query_entities


def test_neighbors_endpoint_logic(corpus_engine: Engine) -> None:
    nb = corpus_engine.neighbors("Helios Sync", depth=1)
    assert nb.found
    related = {e.subject for e in nb.edges} | {e.obj for e in nb.edges}
    assert "Helios Labs" in related
    missing = corpus_engine.neighbors("Unknown Entity Xyz")
    assert not missing.found and missing.edges == []


def test_state_survives_reopen(corpus_engine: Engine) -> None:
    """SQLite + Chroma persist; a new Engine over the same dir answers identically."""
    reopened = Engine(corpus_engine.config, corpus_engine.embedder, corpus_engine.llm)
    try:
        original = corpus_engine.ask("Who owns the Meridian Policy?")
        again = reopened.ask("Who owns the Meridian Policy?")
        assert again.answer == original.answer
        assert again.debug.max_fused_score == original.debug.max_fused_score
    finally:
        pass  # shares the sqlite file with corpus_engine; closed by its fixture
