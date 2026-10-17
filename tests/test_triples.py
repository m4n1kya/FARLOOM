"""FakeLLM triple extraction: determinism and expected output."""

from graphrag_knowledge_engine.providers.llm import FakeLLM, extract_entities


def test_extraction_is_deterministic() -> None:
    llm = FakeLLM()
    text = (
        "Aurora Dynamics acquired Helios Labs in 2023. "
        "Daniel Okafor leads Platform Engineering from Cascade Campus."
    )
    runs = [llm.extract_triples(text) for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert runs[0], "expected at least one triple"


def test_co_occurrence_pairs_within_sentence() -> None:
    triples = FakeLLM().extract_triples("Aurora Dynamics acquired Helios Labs.")
    assert ("Aurora Dynamics", "acquired", "Helios Labs") in triples


def test_relation_falls_back_when_gap_is_long() -> None:
    text = (
        "Pulse Engine handles ordering, deduplication, compaction, batching, "
        "and replay of every stream before Beacon API sees it."
    )
    triples = FakeLLM().extract_triples(text)
    pairs = [(s, o) for s, _r, o in triples]
    assert ("Pulse Engine", "Beacon API") in pairs
    relation = next(r for s, r, o in triples if (s, o) == ("Pulse Engine", "Beacon API"))
    assert relation == "related_to"  # gap longer than 3 content tokens


def test_no_cross_sentence_pairs() -> None:
    triples = FakeLLM().extract_triples("Helios Labs ships software. Lisbon Hub is an office.")
    pairs = [(s, o) for s, _r, o in triples]
    assert ("Helios Labs", "Lisbon Hub") not in pairs


def test_sentence_initial_stopwords_are_not_entities() -> None:
    entities = extract_entities("The Meridian Policy applies to Aurora Dynamics.")
    assert "Meridian Policy" in entities
    assert all(not e.startswith("The ") for e in entities)


def test_entity_order_and_uniqueness() -> None:
    entities = extract_entities("Beacon API talks to Pulse Engine. Beacon API retries.")
    assert entities == ["Beacon API", "Pulse Engine"]
