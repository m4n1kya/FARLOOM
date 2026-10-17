"""Chunking strategies.

Two strategies, selectable per ingest:

* ``RecursiveCharacterChunker`` - splits on a separator hierarchy
  (paragraph -> line -> sentence -> word), then packs pieces into windows of
  ``chunk_size`` characters with ``overlap`` characters carried between
  consecutive chunks. Format-agnostic workhorse.
* ``HeadingAwareChunker`` - splits Markdown-style documents at headings and
  keeps each section intact (recursively splitting only oversized sections),
  recording the heading path as ``section`` metadata. Better retrieval
  granularity for structured docs; the eval harness can quantify by how much.
"""

from __future__ import annotations

import re
from typing import Protocol

from graphrag_knowledge_engine.ingest.loaders import LoadedDocument
from graphrag_knowledge_engine.models import Chunk, ChunkStrategy

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class Chunker(Protocol):
    name: ChunkStrategy

    def chunk(self, doc: LoadedDocument) -> list[Chunk]: ...


class RecursiveCharacterChunker:
    """Recursive character splitter with overlap."""

    name: ChunkStrategy = "recursive"

    def __init__(self, chunk_size: int = 800, overlap: int = 120) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._separators = ["\n\n", "\n", ". ", " "]

    def chunk(self, doc: LoadedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page, text in doc.pages:
            for piece in self.split_text(text):
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}:{len(chunks):04d}",
                        doc_id=doc.doc_id,
                        text=piece,
                        source=doc.source,
                        page=page,
                        section=None,
                        position=len(chunks),
                        strategy=self.name,
                    )
                )
        return chunks

    def split_text(self, text: str) -> list[str]:
        """Split text into windows of <= chunk_size chars with overlap."""
        pieces = self._split_recursive(text, 0)
        return self._pack(pieces)

    def _split_recursive(self, text: str, sep_index: int) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        if sep_index >= len(self._separators):
            return [
                text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)
            ]
        sep = self._separators[sep_index]
        parts = [p for p in _split_keep_suffix(text, sep) if p.strip()]
        if len(parts) <= 1:
            return self._split_recursive(text, sep_index + 1)
        out: list[str] = []
        for part in parts:
            if len(part) > self.chunk_size:
                out.extend(self._split_recursive(part, sep_index + 1))
            else:
                out.append(part)
        return out

    def _pack(self, pieces: list[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            if current and len(current) + len(piece) > self.chunk_size:
                chunks.append(current.strip())
                tail = current[-self.overlap :] if self.overlap else ""
                current = tail.lstrip() + piece
            else:
                current += piece
        if current.strip():
            chunks.append(current.strip())
        return chunks


class HeadingAwareChunker:
    """Structural chunker keyed on Markdown headings."""

    name: ChunkStrategy = "heading"

    def __init__(self, max_section_chars: int = 1600, chunk_size: int = 800) -> None:
        self.max_section_chars = max_section_chars
        self._fallback = RecursiveCharacterChunker(chunk_size=chunk_size, overlap=0)

    def chunk(self, doc: LoadedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        for page, text in doc.pages:
            for section_path, body in self._sections(text):
                bodies = (
                    self._fallback.split_text(body)
                    if len(body) > self.max_section_chars
                    else [body]
                )
                for piece in bodies:
                    if not piece.strip():
                        continue
                    chunks.append(
                        Chunk(
                            chunk_id=f"{doc.doc_id}:{len(chunks):04d}",
                            doc_id=doc.doc_id,
                            text=piece.strip(),
                            source=doc.source,
                            page=page,
                            section=section_path,
                            position=len(chunks),
                            strategy=self.name,
                        )
                    )
        return chunks

    def _sections(self, text: str) -> list[tuple[str | None, str]]:
        """Split into (heading-path, body) pairs. Text before any heading gets None."""
        matches = list(_HEADING_RE.finditer(text))
        if not matches:
            return [(None, text)]
        sections: list[tuple[str | None, str]] = []
        preamble = text[: matches[0].start()]
        if preamble.strip():
            sections.append((None, preamble))
        # Track the active heading at each level to build "H1 > H2" paths.
        path: dict[int, str] = {}
        for i, m in enumerate(matches):
            level = len(m.group(1))
            title = m.group(2).strip()
            path[level] = title
            for deeper in [lv for lv in path if lv > level]:
                del path[deeper]
            section_path = " > ".join(path[lv] for lv in sorted(path))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[m.end() : end]
            if body.strip():
                # Keep a blank line between title and body so paragraph-aware
                # consumers (entity extraction) never join them into one sentence.
                sections.append((section_path, f"{title}\n\n{body.strip()}"))
        return sections


def get_chunker(
    strategy: ChunkStrategy, chunk_size: int = 800, overlap: int = 120
) -> Chunker:
    if strategy == "heading":
        return HeadingAwareChunker(chunk_size=chunk_size)
    return RecursiveCharacterChunker(chunk_size=chunk_size, overlap=overlap)


def _split_keep_suffix(text: str, sep: str) -> list[str]:
    """Split on `sep` but keep the separator attached to the preceding part."""
    parts = text.split(sep)
    return [part + sep for part in parts[:-1]] + [parts[-1]]
