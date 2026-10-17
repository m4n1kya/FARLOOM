"""Shared fixtures: an offline engine over the bundled corpus."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from graphrag_knowledge_engine.config import EngineConfig
from graphrag_knowledge_engine.engine import Engine, create_engine

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "data" / "corpus"
GOLD_PATH = ROOT / "data" / "gold" / "qa_pairs.json"


@pytest.fixture(scope="session")
def corpus_engine(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Engine]:
    """Engine with the full bundled corpus ingested (fake providers, offline)."""
    data_dir = tmp_path_factory.mktemp("engine")
    engine = create_engine(EngineConfig(data_dir=data_dir, provider="fake"))
    for doc in sorted(CORPUS_DIR.glob("*.md")):
        engine.ingest_path(doc, strategy="heading")
    yield engine
    engine.close()


@pytest.fixture()
def empty_engine(tmp_path: Path) -> Iterator[Engine]:
    engine = create_engine(EngineConfig(data_dir=tmp_path / "engine", provider="fake"))
    yield engine
    engine.close()
