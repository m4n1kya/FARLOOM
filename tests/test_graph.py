"""Knowledge graph: bounded expansion, mention voting, neighbor queries."""

from graphrag_knowledge_engine.models import Triple
from graphrag_knowledge_engine.store.graph import KnowledgeGraph, normalize_entity


def _t(s: str, r: str, o: str, chunk: str) -> Triple:
    return Triple(subject=s, relation=r, obj=o, chunk_id=chunk, doc_id="d")


def chain_graph() -> KnowledgeGraph:
    # A -> B -> C -> D -> E, one chunk per edge.
    return KnowledgeGraph.from_triples(
        [
            _t("A1", "r", "B1", "c_ab"),
            _t("B1", "r", "C1", "c_bc"),
            _t("C1", "r", "D1", "c_cd"),
            _t("D1", "r", "E1", "c_de"),
        ]
    )


def test_expansion_respects_depth_bound() -> None:
    kg = chain_graph()
    chunks = dict(kg.expand(["A1"], max_depth=2, max_nodes=100, top_k=10))
    # Depth-2 frontier = {A1, B1, C1}. c_cd is mentioned by C1 (hop 2) so it
    # is reachable, but c_de involves only D1/E1 (hop 3+) and must not be.
    assert set(chunks) == {"c_ab", "c_bc", "c_cd"}
    assert "c_de" not in chunks


def test_expansion_scores_prefer_near_seed() -> None:
    kg = chain_graph()
    chunks = dict(kg.expand(["A1"], max_depth=2, max_nodes=100, top_k=10))
    assert chunks["c_ab"] > chunks["c_bc"] > chunks["c_cd"]


def test_expansion_respects_max_nodes() -> None:
    kg = chain_graph()
    # max_nodes=2 stops BFS after {A1, B1}: C1 never votes, so c_cd is gone.
    chunks = dict(kg.expand(["A1"], max_depth=2, max_nodes=2, top_k=10))
    assert set(chunks) == {"c_ab", "c_bc"}


def test_expansion_respects_top_k() -> None:
    star = KnowledgeGraph.from_triples(
        [_t("Hub1", "r", f"Leaf{i}", f"c{i:02d}") for i in range(50)]
    )
    assert len(star.expand(["Hub1"], max_depth=2, max_nodes=100, top_k=7)) == 7


def test_mentions_without_triples_still_vote() -> None:
    kg = chain_graph()
    kg.add_mentions([("A1", "c_solo")])
    chunks = dict(kg.expand(["A1"], max_depth=2, max_nodes=100, top_k=10))
    # c_solo mentions only the seed: full closeness, maximal focus.
    assert chunks["c_solo"] >= chunks["c_ab"]


def test_focus_penalizes_hub_chunks() -> None:
    kg = KnowledgeGraph()
    kg.add_mentions([("Seed1", "c_focused")])
    kg.add_mentions([("Seed1", "c_hub"), ("Other1", "c_hub"), ("Other2", "c_hub")])
    ranked = kg.expand(["Seed1"], max_depth=2, max_nodes=100, top_k=10)
    assert [cid for cid, _ in ranked] == ["c_focused", "c_hub"]


def test_expansion_unknown_seed_is_empty() -> None:
    assert chain_graph().expand(["Nope"], max_depth=2, max_nodes=10, top_k=5) == []


def test_provenance_merged_for_duplicate_edges() -> None:
    kg = KnowledgeGraph.from_triples(
        [_t("X1", "uses", "Y1", "c1"), _t("X1", "uses", "Y1", "c2")]
    )
    assert kg.num_edges == 1
    edges = kg.neighbors("X1")
    assert edges[0].chunk_ids == ["c1", "c2"]


def test_detect_query_entities_longest_match_wins() -> None:
    kg = KnowledgeGraph.from_triples(
        [_t("Helios", "r", "Other1", "c1"), _t("Helios Sync", "r", "Other2", "c2")]
    )
    found = kg.detect_query_entities("How reliable is Helios Sync today?")
    assert found == ["Helios Sync"]


def test_normalize_entity() -> None:
    assert normalize_entity("  Beacon   API ") == "beacon api"
