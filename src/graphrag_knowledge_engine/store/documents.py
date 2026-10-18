"""SQLite-backed store for documents, chunks, and extracted triples.

One file (``store.sqlite``) holds everything relational: chunk text +
metadata for BM25 rebuilds and citation rendering, and the triple table that
the knowledge graph is hydrated from at startup.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from graphrag_knowledge_engine.models import Chunk, ChunkStrategy, Document, Triple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    fmt TEXT NOT NULL,
    num_chunks INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    page INTEGER,
    section TEXT,
    position INTEGER NOT NULL,
    strategy TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS triples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    obj TEXT NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    doc_id TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mentions (
    entity TEXT NOT NULL,
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    UNIQUE(entity, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_triples_chunk ON triples(chunk_id);
CREATE INDEX IF NOT EXISTS idx_mentions_chunk ON mentions(chunk_id);
"""


class DocumentStore:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: FastAPI serves sync endpoints from a thread
        # pool, so the connection outlives the thread that created it. Access
        # is single-writer in practice (one engine instance per process).
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- documents / chunks ------------------------------------------------

    def upsert_document(self, doc: Document, chunks: list[Chunk]) -> None:
        """Replace a document and all of its chunks/triples atomically."""
        with self._conn:
            self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc.doc_id,))
            self._conn.execute(
                "INSERT INTO documents (doc_id, source, title, fmt, num_chunks) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc.doc_id, doc.source, doc.title, doc.fmt, doc.num_chunks),
            )
            self._conn.executemany(
                "INSERT INTO chunks "
                "(chunk_id, doc_id, text, source, page, section, position, strategy) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (c.chunk_id, c.doc_id, c.text, c.source, c.page, c.section, c.position,
                     c.strategy)
                    for c in chunks
                ],
            )

    def all_chunks(self) -> list[Chunk]:
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, text, source, page, section, position, strategy "
            "FROM chunks ORDER BY doc_id, position"
        ).fetchall()
        return [_row_to_chunk(row) for row in rows]

    def get_chunks(self, chunk_ids: list[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._conn.execute(
            "SELECT chunk_id, doc_id, text, source, page, section, position, strategy "
            f"FROM chunks WHERE chunk_id IN ({placeholders})",
            chunk_ids,
        ).fetchall()
        return {row[0]: _row_to_chunk(row) for row in rows}

    def list_documents(self) -> list[Document]:
        rows = self._conn.execute(
            "SELECT doc_id, source, title, fmt, num_chunks FROM documents ORDER BY source"
        ).fetchall()
        return [
            Document(doc_id=r[0], source=r[1], title=r[2], fmt=r[3], num_chunks=r[4])
            for r in rows
        ]

    def get_document(self, doc_id_or_source: str) -> Document | None:
        """Find a document by doc_id or source path."""
        row = self._conn.execute(
            "SELECT doc_id, source, title, fmt, num_chunks FROM documents WHERE doc_id = ? OR source = ?",
            (doc_id_or_source, doc_id_or_source),
        ).fetchone()
        if not row:
            return None
        return Document(doc_id=row[0], source=row[1], title=row[2], fmt=row[3], num_chunks=row[4])

    def get_chunk_ids_for_doc(self, doc_id_or_source: str) -> list[str]:
        """Return all chunk IDs belonging to a document."""
        rows = self._conn.execute(
            "SELECT chunk_id FROM chunks WHERE doc_id = ? OR source = ?",
            (doc_id_or_source, doc_id_or_source),
        ).fetchall()
        return [r[0] for r in rows]

    def delete_document(self, doc_id_or_source: str) -> bool:
        """Atomically delete a document and cascade-delete its chunks, triples, and mentions."""
        with self._conn:
            res = self._conn.execute(
                "DELETE FROM documents WHERE doc_id = ? OR source = ?",
                (doc_id_or_source, doc_id_or_source),
            )
            return res.rowcount > 0

    # -- triples -------------------------------------------------------------

    def add_triples(self, triples: list[Triple]) -> None:
        with self._conn:
            self._conn.executemany(
                "INSERT INTO triples (subject, relation, obj, chunk_id, doc_id) "
                "VALUES (?, ?, ?, ?, ?)",
                [(t.subject, t.relation, t.obj, t.chunk_id, t.doc_id) for t in triples],
            )

    def all_triples(self) -> list[Triple]:
        rows = self._conn.execute(
            "SELECT subject, relation, obj, chunk_id, doc_id FROM triples ORDER BY id"
        ).fetchall()
        return [
            Triple(subject=r[0], relation=r[1], obj=r[2], chunk_id=r[3], doc_id=r[4])
            for r in rows
        ]

    # -- entity mentions -----------------------------------------------------

    def add_mentions(self, mentions: list[tuple[str, str]]) -> None:
        """Record (entity display name, chunk_id) mention provenance."""
        with self._conn:
            self._conn.executemany(
                "INSERT OR IGNORE INTO mentions (entity, chunk_id) VALUES (?, ?)", mentions
            )

    def all_mentions(self) -> list[tuple[str, str]]:
        rows = self._conn.execute(
            "SELECT entity, chunk_id FROM mentions ORDER BY entity, chunk_id"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]


def _row_to_chunk(row: tuple[str, str, str, str, int | None, str | None, int, str]) -> Chunk:
    strategy: ChunkStrategy = "heading" if row[7] == "heading" else "recursive"
    return Chunk(
        chunk_id=row[0],
        doc_id=row[1],
        text=row[2],
        source=row[3],
        page=row[4],
        section=row[5],
        position=row[6],
        strategy=strategy,
    )
