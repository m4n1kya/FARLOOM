"""File loaders: format detection, page handling, and a real PDF round-trip.

The PDF fixture is generated in-test as raw PDF bytes (one page, one text
object) so the repository stays text-only and the test still exercises the
real pypdf extraction path.
"""

from pathlib import Path

import pytest

from graphrag_knowledge_engine.ingest.loaders import load_document


def _minimal_pdf(text: str) -> bytes:
    """Build a minimal one-page PDF containing `text` with a valid xref table."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def test_load_markdown(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nBody text here.\n", encoding="utf-8")
    doc = load_document(p)
    assert doc.fmt == "markdown"
    assert doc.title == "Title"
    assert doc.pages == [(None, "# Title\n\nBody text here.\n")]


def test_load_text_title_falls_back_to_first_line(tmp_path: Path) -> None:
    p = tmp_path / "notes.txt"
    p.write_text("Plain first line.\nSecond line.\n", encoding="utf-8")
    doc = load_document(p)
    assert doc.fmt == "text"
    assert doc.title == "Plain first line."


def test_load_pdf_extracts_pages(tmp_path: Path) -> None:
    p = tmp_path / "doc.pdf"
    p.write_bytes(_minimal_pdf("Helios Sync runs on HX-9 nodes."))
    doc = load_document(p)
    assert doc.fmt == "pdf"
    assert len(doc.pages) == 1
    page_number, text = doc.pages[0]
    assert page_number == 1
    assert "HX-9" in text


def test_doc_id_is_stable_for_same_content(tmp_path: Path) -> None:
    p = tmp_path / "a.md"
    p.write_text("# Same\n\ncontent\n", encoding="utf-8")
    assert load_document(p).doc_id == load_document(p).doc_id


def test_unknown_extension_rejected(tmp_path: Path) -> None:
    p = tmp_path / "doc.docx"
    p.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(p)


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "missing.md")
