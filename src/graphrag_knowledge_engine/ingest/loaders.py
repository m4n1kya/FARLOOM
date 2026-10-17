"""File loaders: PDF (pypdf), Markdown, and plain text.

Loaders normalise every format into a list of (page_number, text) pairs so the
chunkers never care where the text came from. Page numbers are 1-based for
PDFs and ``None`` for text formats.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pypdf import PdfReader

DocFormat = Literal["pdf", "markdown", "text"]

_SUFFIX_TO_FORMAT: dict[str, DocFormat] = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
}


class LoadedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    title: str
    fmt: DocFormat
    pages: list[tuple[int | None, str]]  # (1-based page or None, text)


def load_document(path: Path) -> LoadedDocument:
    """Load a PDF/Markdown/TXT file into a normalised document."""
    suffix = path.suffix.lower()
    fmt = _SUFFIX_TO_FORMAT.get(suffix)
    if fmt is None:
        raise ValueError(f"Unsupported file type: {suffix!r} (expected .pdf, .md, .txt)")
    if not path.is_file():
        raise FileNotFoundError(f"No such file: {path}")

    if fmt == "pdf":
        reader = PdfReader(str(path))
        pages: list[tuple[int | None, str]] = [
            (i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)
        ]
        raw = "".join(text for _, text in pages)
    else:
        raw = path.read_text(encoding="utf-8")
        pages = [(None, raw)]

    doc_id = hashlib.sha256((path.name + raw).encode("utf-8")).hexdigest()[:16]
    return LoadedDocument(
        doc_id=doc_id,
        source=path.name,
        title=_derive_title(raw, fallback=path.stem),
        fmt=fmt,
        pages=pages,
    )


def _derive_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
        if stripped:
            return stripped[:80]
    return fallback
