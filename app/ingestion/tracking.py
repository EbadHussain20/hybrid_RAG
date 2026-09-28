"""SQLite-backed document tracking store.

Why SQLite: it needs zero external services, persists to a single file,
supports concurrent reads safely, and is more than sufficient for tracking a
few thousand internal documents' hashes/metadata. This is what enables
incremental indexing: we skip re-embedding documents whose content hash is
unchanged since the last run.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from app.core.config import settings

DB_PATH = Path(settings.INDEX_DIR) / "documents.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    document_title TEXT NOT NULL,
    source_path TEXT NOT NULL,
    document_type TEXT NOT NULL,
    department TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH))
    try:
        connection.execute(_SCHEMA)
        yield connection
        connection.commit()
    finally:
        connection.close()


def get_known_hash(document_id: str) -> str | None:
    with _conn() as c:
        row = c.execute(
            "SELECT content_hash FROM documents WHERE document_id = ?", (document_id,)
        ).fetchone()
        return row[0] if row else None


def upsert_document(document_id: str, title: str, source_path: str, document_type: str,
                     department: str, content_hash: str, created_at: str, updated_at: str) -> None:
    with _conn() as c:
        c.execute(
            """
            INSERT INTO documents (document_id, document_title, source_path, document_type,
                                    department, content_hash, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET
                document_title=excluded.document_title,
                source_path=excluded.source_path,
                document_type=excluded.document_type,
                department=excluded.department,
                content_hash=excluded.content_hash,
                updated_at=excluded.updated_at
            """,
            (document_id, title, source_path, document_type, department, content_hash, created_at, updated_at),
        )


def list_documents() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT document_id, document_title, source_path, document_type, department, "
            "content_hash, created_at, updated_at FROM documents ORDER BY updated_at DESC"
        ).fetchall()
        cols = ["document_id", "document_title", "source_path", "document_type",
                "department", "content_hash", "created_at", "updated_at"]
        return [dict(zip(cols, row)) for row in rows]


def get_document(document_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT document_id, document_title, source_path, document_type, department, "
            "content_hash, created_at, updated_at FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        if not row:
            return None
        cols = ["document_id", "document_title", "source_path", "document_type",
                "department", "content_hash", "created_at", "updated_at"]
        return dict(zip(cols, row))


def delete_document(document_id: str) -> None:
    with _conn() as c:
        c.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
