"""Postgres registry of ingested PDFs, keyed by the SHA-256 of the file bytes."""
from typing import List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import DATABASE_URL, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# A 'processing' row older than this is treated as a crashed ingestion and can be retried
STALE_PROCESSING = "30 minutes"

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id              TEXT PRIMARY KEY,
    filename        TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('processing', 'done', 'failed')),
    chunk_count     INT,
    embedding_model TEXT NOT NULL,
    chunk_size      INT NOT NULL,
    chunk_overlap   INT NOT NULL,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def get_connection() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)


def init_db() -> None:
    with get_connection() as conn:
        conn.execute(SCHEMA)


def claim_document(doc_id: str, filename: str) -> bool:
    """Mark a document as processing. Returns False if it is already indexed
    with the current settings, or another request is processing it."""
    with get_connection() as conn:
        row = conn.execute(
            """
            INSERT INTO documents (id, filename, status, embedding_model, chunk_size, chunk_overlap)
            VALUES (%(id)s, %(filename)s, 'processing', %(model)s, %(size)s, %(overlap)s)
            ON CONFLICT (id) DO UPDATE SET
                filename = EXCLUDED.filename,
                status = 'processing',
                embedding_model = EXCLUDED.embedding_model,
                chunk_size = EXCLUDED.chunk_size,
                chunk_overlap = EXCLUDED.chunk_overlap,
                error = NULL,
                updated_at = now()
            WHERE NOT (
                -- already indexed with the current settings
                (documents.status = 'done'
                 AND documents.embedding_model = EXCLUDED.embedding_model
                 AND documents.chunk_size = EXCLUDED.chunk_size
                 AND documents.chunk_overlap = EXCLUDED.chunk_overlap)
                -- another request is ingesting it right now
                OR (documents.status = 'processing'
                    AND documents.updated_at > now() - %(stale)s::interval)
            )
            RETURNING id
            """,
            {
                "id": doc_id,
                "filename": filename,
                "model": EMBEDDING_MODEL,
                "size": CHUNK_SIZE,
                "overlap": CHUNK_OVERLAP,
                "stale": STALE_PROCESSING,
            },
        ).fetchone()
    return row is not None


def mark_done(doc_id: str, chunk_count: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE documents SET status = 'done', chunk_count = %s, updated_at = now() WHERE id = %s",
            (chunk_count, doc_id),
        )


def mark_failed(doc_id: str, error: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE documents SET status = 'failed', error = %s, updated_at = now() WHERE id = %s",
            (error, doc_id),
        )


def get_document(doc_id: str) -> Optional[dict]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM documents WHERE id = %s", (doc_id,)).fetchone()


def list_documents() -> List[dict]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()


def delete_document(doc_id: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
