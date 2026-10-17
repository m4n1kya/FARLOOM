"""Chunker behaviour: boundaries, overlap, and metadata."""

from graphrag_knowledge_engine.ingest.chunkers import (
    HeadingAwareChunker,
    RecursiveCharacterChunker,
)
from graphrag_knowledge_engine.ingest.loaders import LoadedDocument


def _doc(text: str) -> LoadedDocument:
    return LoadedDocument(
        doc_id="doc1", source="doc.md", title="Doc", fmt="markdown", pages=[(None, text)]
    )


def test_recursive_respects_chunk_size() -> None:
    text = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(8))
    chunker = RecursiveCharacterChunker(chunk_size=300, overlap=50)
    pieces = chunker.split_text(text)
    assert len(pieces) > 1
    # Packing may join a tail overlap with one oversized piece, hence the slack.
    assert all(len(p) <= 300 + 50 for p in pieces)


def test_recursive_overlap_carries_text_between_chunks() -> None:
    text = " ".join(f"w{i:03d}" for i in range(200))  # no paragraph/sentence breaks
    chunker = RecursiveCharacterChunker(chunk_size=200, overlap=60)
    pieces = chunker.split_text(text)
    assert len(pieces) >= 2
    for prev, nxt in zip(pieces, pieces[1:], strict=False):
        tail_tokens = prev[-60:].split()[1:]  # drop a possibly clipped first token
        assert tail_tokens, "expected a non-empty overlap window"
        assert " ".join(tail_tokens) in nxt


def test_recursive_chunk_metadata_and_positions() -> None:
    chunks = RecursiveCharacterChunker(chunk_size=120, overlap=20).chunk(
        _doc("First paragraph here.\n\nSecond paragraph follows with more words in it.")
    )
    assert [c.position for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.chunk_id == f"doc1:{c.position:04d}"
        assert c.source == "doc.md"
        assert c.page is None
        assert c.section is None
        assert c.strategy == "recursive"


def test_heading_chunker_sections_and_paths() -> None:
    text = (
        "Intro before any heading.\n\n"
        "# Title\n\nTop-level body.\n\n"
        "## Alpha\n\nAlpha body text.\n\n"
        "### Alpha Detail\n\nNested body.\n\n"
        "## Beta\n\nBeta body text.\n"
    )
    chunks = HeadingAwareChunker().chunk(_doc(text))
    sections = [c.section for c in chunks]
    assert sections[0] is None  # preamble
    assert "Title" in sections
    assert "Title > Alpha" in sections
    assert "Title > Alpha > Alpha Detail" in sections
    assert "Title > Beta" in sections  # sibling resets the deeper level
    assert all(c.strategy == "heading" for c in chunks)


def test_heading_chunker_splits_oversized_sections() -> None:
    body = "word " * 800
    chunks = HeadingAwareChunker(max_section_chars=400, chunk_size=300).chunk(
        _doc(f"# Big\n\n{body}")
    )
    assert len(chunks) > 1
    assert all(c.section == "Big" for c in chunks)


def test_pdf_page_metadata_is_preserved() -> None:
    doc = LoadedDocument(
        doc_id="p", source="x.pdf", title="X", fmt="pdf",
        pages=[(1, "Page one text."), (2, "Page two text.")],
    )
    chunks = RecursiveCharacterChunker(chunk_size=500, overlap=50).chunk(doc)
    assert [c.page for c in chunks] == [1, 2]
