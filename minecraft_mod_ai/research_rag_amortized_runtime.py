from __future__ import annotations

"""Amortized helpers for the canonical incremental Project-RAG owner.

`research_rag_performance` owns the incremental build algorithm.  This module no
longer carries a second copy of that algorithm; it only replaces the narrow helpers
that remove migration-wide hashing and repeated semantic-LSH scans.  Content deletion
invalidates the LSH readiness token before delegating to the canonical delete helper.
"""

import sqlite3
from functools import wraps
from typing import Any

from .research_perf_common import env_int, table_exists

_MARKER = "_mmm_research_rag_amortized_v2"


def _bootstrap_incremental_state(connection: sqlite3.Connection) -> None:
    count = int(connection.execute("SELECT COUNT(*) FROM mmm_file_state").fetchone()[0])
    if count:
        return
    rows = connection.execute(
        "SELECT source_path, size_bytes, modified_ns FROM indexed_files ORDER BY source_path"
    ).fetchall()
    connection.executemany(
        """
        INSERT OR REPLACE INTO mmm_file_state(
            source_path, size_bytes, modified_ns, content_sha256
        ) VALUES (?, ?, ?, '')
        """,
        [(str(path), int(size), int(mtime)) for path, size, mtime in rows],
    )


def _lsh_index_token(connection: sqlite3.Connection) -> str:
    if not table_exists(connection, "index_meta"):
        return ""
    row = connection.execute(
        "SELECT value FROM index_meta WHERE key = 'chunks_indexed'"
    ).fetchone()
    return str(row[0]) if row is not None else ""


def _lsh_ready_token(connection: sqlite3.Connection) -> str:
    if not table_exists(connection, "index_meta"):
        return ""
    row = connection.execute(
        "SELECT value FROM index_meta WHERE key = 'mmm_semantic_lsh_chunks'"
    ).fetchone()
    return str(row[0]) if row is not None else ""


def _invalidate_semantic_lsh(connection: sqlite3.Connection) -> None:
    if table_exists(connection, "index_meta"):
        connection.execute(
            "DELETE FROM index_meta WHERE key IN "
            "('mmm_semantic_lsh_chunks', 'mmm_semantic_lsh_valid')"
        )


def _ensure_semantic_lsh(perf: Any, connection: sqlite3.Connection) -> None:
    token = _lsh_index_token(connection)
    if (
        token
        and table_exists(connection, "mmm_semantic_lsh")
        and _lsh_ready_token(connection) == token
    ):
        return

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS mmm_semantic_lsh (
            chunk_id TEXT PRIMARY KEY,
            sig_a INTEGER NOT NULL,
            sig_b INTEGER NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS mmm_semantic_lsh_a ON mmm_semantic_lsh(sig_a)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS mmm_semantic_lsh_b ON mmm_semantic_lsh(sig_b)"
    )
    connection.execute(
        "DELETE FROM mmm_semantic_lsh WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks)"
    )

    batch_size = env_int("MMM_RAG_LSH_BUILD_BATCH", 256, minimum=32, maximum=2048)
    while True:
        rows = connection.execute(
            """
            SELECT c.chunk_id, c.embedding
            FROM chunks AS c
            LEFT JOIN mmm_semantic_lsh AS l ON l.chunk_id = c.chunk_id
            WHERE l.chunk_id IS NULL AND c.embedding != '[]'
            ORDER BY c.id
            LIMIT ?
            """,
            (batch_size,),
        ).fetchall()
        if not rows:
            break
        valid_ids: list[str] = []
        vectors: list[list[float]] = []
        for row in rows:
            vector = perf._parse_embedding(row[1])
            if vector:
                valid_ids.append(str(row[0]))
                vectors.append(vector)
        if valid_ids:
            signatures = perf._signatures(vectors)
            connection.executemany(
                "INSERT OR REPLACE INTO mmm_semantic_lsh"
                "(chunk_id, sig_a, sig_b) VALUES (?, ?, ?)",
                [
                    (chunk_id, sig_a, sig_b)
                    for chunk_id, (sig_a, sig_b) in zip(
                        valid_ids,
                        signatures,
                        strict=True,
                    )
                ],
            )
        valid_set = set(valid_ids)
        invalid_ids = [str(row[0]) for row in rows if str(row[0]) not in valid_set]
        if invalid_ids:
            connection.executemany(
                "INSERT OR REPLACE INTO mmm_semantic_lsh"
                "(chunk_id, sig_a, sig_b) VALUES (?, -1, -1)",
                [(chunk_id,) for chunk_id in invalid_ids],
            )

    if token and table_exists(connection, "index_meta"):
        valid_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM mmm_semantic_lsh WHERE sig_a >= 0 OR sig_b >= 0"
            ).fetchone()[0]
        )
        connection.execute(
            """
            INSERT INTO index_meta(key, value) VALUES ('mmm_semantic_lsh_chunks', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (token,),
        )
        connection.execute(
            """
            INSERT INTO index_meta(key, value) VALUES ('mmm_semantic_lsh_valid', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(valid_count),),
        )
    connection.commit()


def harden(_rag_index_module: Any, perf_module: Any) -> None:
    """Patch helpers used by the existing incremental build; never replace build."""

    if getattr(perf_module._ensure_semantic_lsh, _MARKER, False):
        return

    perf_module._bootstrap_incremental_state = _bootstrap_incremental_state

    current_delete = perf_module._delete_rag_path
    if not getattr(current_delete, _MARKER, False):

        @wraps(current_delete)
        def delete_rag_path(
            connection: sqlite3.Connection,
            source_path: str,
            *,
            fts5: bool,
        ) -> None:
            _invalidate_semantic_lsh(connection)
            current_delete(connection, source_path, fts5=fts5)

        setattr(delete_rag_path, _MARKER, True)
        delete_rag_path.__wrapped__ = current_delete  # type: ignore[attr-defined]
        perf_module._delete_rag_path = delete_rag_path

    def ensure(connection: sqlite3.Connection) -> None:
        _ensure_semantic_lsh(perf_module, connection)

    setattr(ensure, _MARKER, True)
    perf_module._ensure_semantic_lsh = ensure


__all__ = ["harden"]
