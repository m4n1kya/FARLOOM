"""Eval harness: gold matching, runner output, and the headline result."""

from pathlib import Path

import pytest

from graphrag_knowledge_engine.evals.runner import (
    GoldEvidence,
    format_table,
    load_gold,
    run_eval,
)
from graphrag_knowledge_engine.models import Chunk, EvalReport
from tests.conftest import CORPUS_DIR, GOLD_PATH


def _chunk(text: str, source: str = "doc.md") -> Chunk:
    return Chunk(
        chunk_id="c", doc_id="d", text=text, source=source,
        page=None, section=None, position=0, strategy="heading",
    )


def test_gold_evidence_matching_is_casefolded_and_doc_scoped() -> None:
    ev = GoldEvidence(doc="doc.md", must_contain="90 Days")
    assert ev.matches(_chunk("logs are retained for 90 days."))
    assert not ev.matches(_chunk("logs are retained for 90 days.", source="other.md"))
    assert not ev.matches(_chunk("logs are retained for 30 days."))


def test_gold_dataset_loads_and_is_well_formed() -> None:
    gold = load_gold(GOLD_PATH)
    assert len(gold.questions) >= 25
    assert all(q.evidence for q in gold.questions)
    assert sum(1 for q in gold.questions if q.kind == "multi-hop") >= 4


@pytest.fixture(scope="module")
def report() -> EvalReport:
    return run_eval(CORPUS_DIR, GOLD_PATH, k=5)


def test_eval_runs_all_four_modes(report: EvalReport) -> None:
    assert [r.mode for r in report.results] == ["dense", "bm25", "hybrid", "hybrid_graph"]
    for r in report.results:
        assert r.answered + r.refused == r.total == report.num_questions
        assert 0.0 <= r.hit_at_k <= 1.0
        assert 0.0 <= r.mrr <= 1.0


def test_hybrid_graph_wins_on_bundled_set(report: EvalReport) -> None:
    """The README headline: hybrid+graph beats every other config here."""
    by_mode = {r.mode: r for r in report.results}
    hg = by_mode["hybrid_graph"]
    for mode in ("dense", "bm25", "hybrid"):
        assert hg.hit_at_k >= by_mode[mode].hit_at_k
        assert hg.mrr >= by_mode[mode].mrr
    assert hg.hit_at_k > by_mode["hybrid"].hit_at_k or hg.mrr > by_mode["hybrid"].mrr


def test_format_table_contains_all_modes(report: EvalReport) -> None:
    table = format_table(report)
    for mode in ("dense", "bm25", "hybrid", "hybrid_graph"):
        assert mode in table
    assert "hit@5" in table and "MRR" in table


def test_eval_requires_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        run_eval(tmp_path, GOLD_PATH, k=5)
