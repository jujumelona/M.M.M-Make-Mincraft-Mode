from __future__ import annotations

"""Amortized hardening for the research-backed Project-RAG runtime.

This layer removes second-order scans left by the first research RAG contract while
preserving its public behavior and exact fallbacks. Existing indexes migrate from
path/size/mtime state without hashing every source file, LSH readiness is invalidated
explicitly on content mutations, and SQLite whole-database checking is opt-in on the
incremental hot path.
"""

import hashlib
import json
import sqlite3
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Sequence

from .research_perf_common import env_bool, env_int, table_exists

_MARKER = "_mmm_research_rag_amortized_v1"


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
            "DELETE FROM index_meta WHERE key IN ('mmm_semantic_lsh_chunks', 'mmm_semantic_lsh_valid')"
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
                "INSERT OR REPLACE INTO mmm_semantic_lsh(chunk_id, sig_a, sig_b) VALUES (?, ?, ?)",
                [
                    (chunk_id, sig_a, sig_b)
                    for chunk_id, (sig_a, sig_b) in zip(valid_ids, signatures, strict=True)
                ],
            )
        valid_set = set(valid_ids)
        invalid_ids = [str(row[0]) for row in rows if str(row[0]) not in valid_set]
        if invalid_ids:
            connection.executemany(
                "INSERT OR REPLACE INTO mmm_semantic_lsh(chunk_id, sig_a, sig_b) VALUES (?, -1, -1)",
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


def _canonical_build(current: Callable[..., Any]) -> Callable[..., Any]:
    candidate = current
    seen: set[int] = set()
    while getattr(candidate, "_mmm_research_rag_performance_v1", False):
        if id(candidate) in seen:
            break
        seen.add(id(candidate))
        wrapped = getattr(candidate, "__wrapped__", None)
        if not callable(wrapped):
            break
        candidate = wrapped
    return candidate


def _incremental_build_factory(rag: Any, perf: Any, current: Callable[..., dict[str, Any]]):
    original = _canonical_build(current)

    @wraps(original)
    def build(
        self: Any,
        roots: Sequence[str | Path],
        *,
        metadata: dict[str, Any],
        router: Any | None = None,
        semantic: bool = False,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        rag._validate_metadata(metadata)
        if semantic and router is None:
            raise ValueError("semantic=True requires a ModelRouter.")
        target = Path(self.index_path).expanduser().resolve()

        def rebuild() -> dict[str, Any]:
            result = original(
                self,
                roots,
                metadata=metadata,
                router=router,
                semantic=semantic,
                max_files=max_files,
            )
            if semantic:
                try:
                    with sqlite3.connect(str(target)) as side_connection:
                        _ensure_semantic_lsh(perf, side_connection)
                except Exception:
                    pass
            return result

        if not target.is_file() or not rag._is_sqlite(target):
            return rebuild()

        with perf._path_lock(target):
            connection: sqlite3.Connection | None = None
            try:
                connection = sqlite3.connect(str(target), timeout=30.0)
                connection.row_factory = sqlite3.Row
                meta = rag._read_index_meta(connection)
                if meta.get("schema_version") != self.schema_version:
                    connection.close()
                    connection = None
                    return rebuild()
                semantic_existing = meta.get("semantic_embeddings") == "1"
                if semantic_existing != bool(semantic):
                    connection.close()
                    connection = None
                    return rebuild()
                try:
                    previous_metadata = json.loads(meta.get("metadata", "{}"))
                except json.JSONDecodeError:
                    previous_metadata = {}
                if perf._stable_rag_metadata(previous_metadata) != perf._stable_rag_metadata(metadata):
                    connection.close()
                    connection = None
                    return rebuild()

                current_paths = list(rag._iter_files(roots, max_files=max_files))
                current_stats: dict[str, tuple[int, int, Path]] = {}
                for path in current_paths:
                    stat = path.stat()
                    current_stats[str(path)] = (
                        int(stat.st_size),
                        int(stat.st_mtime_ns),
                        path,
                    )

                perf._initialize_incremental_state(connection)
                _bootstrap_incremental_state(connection)
                connection.commit()
                previous = {
                    str(row[0]): (int(row[1]), int(row[2]), str(row[3]))
                    for row in connection.execute(
                        "SELECT source_path, size_bytes, modified_ns, content_sha256 FROM mmm_file_state"
                    )
                }
                removed = sorted(set(previous) - set(current_stats))
                changed: list[tuple[str, Path, int, int, str]] = []
                metadata_only: list[tuple[str, int, int, str]] = []
                for source_path, (size_bytes, modified_ns, path) in current_stats.items():
                    old = previous.get(source_path)
                    if old is not None and old[0] == size_bytes and old[1] == modified_ns:
                        continue
                    digest = perf._file_sha256(path)
                    if old is not None and old[2] and old[2] == digest:
                        metadata_only.append((source_path, size_bytes, modified_ns, digest))
                    else:
                        changed.append((source_path, path, size_bytes, modified_ns, digest))

                fts5 = meta.get("fts5") == "1"
                connection.execute("BEGIN IMMEDIATE")
                try:
                    if semantic and (removed or changed):
                        _invalidate_semantic_lsh(connection)
                    for source_path in removed:
                        perf._delete_rag_path(connection, source_path, fts5=fts5)
                    for source_path, _path, _size, _mtime, _digest in changed:
                        perf._delete_rag_path(connection, source_path, fts5=fts5)
                    for source_path, size_bytes, modified_ns, digest in metadata_only:
                        connection.execute(
                            "UPDATE indexed_files SET size_bytes = ?, modified_ns = ? WHERE source_path = ?",
                            (size_bytes, modified_ns, source_path),
                        )
                        connection.execute(
                            """
                            INSERT INTO mmm_file_state(source_path, size_bytes, modified_ns, content_sha256)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(source_path) DO UPDATE SET
                                size_bytes = excluded.size_bytes,
                                modified_ns = excluded.modified_ns,
                                content_sha256 = excluded.content_sha256
                            """,
                            (source_path, size_bytes, modified_ns, digest),
                        )

                    embedding_dimensions = int(meta.get("embedding_dimensions", "0") or 0) or None
                    batch: list[Any] = []
                    batch_size = rag._EMBEDDING_BATCH_SIZE if semantic else rag._INSERT_BATCH_SIZE

                    def flush_batch() -> None:
                        nonlocal embedding_dimensions
                        if not batch:
                            return
                        _inserted, dimension = rag._insert_chunk_batch(
                            connection,
                            batch,
                            fts5_available=fts5,
                            router=router,
                            semantic=semantic,
                            expected_embedding_dimension=embedding_dimensions,
                        )
                        if dimension is not None:
                            embedding_dimensions = dimension
                        batch.clear()

                    for source_path, path, size_bytes, modified_ns, digest in changed:
                        connection.execute(
                            "INSERT INTO indexed_files(source_path, size_bytes, modified_ns) VALUES (?, ?, ?)",
                            (source_path, size_bytes, modified_ns),
                        )
                        connection.execute(
                            "INSERT INTO mmm_file_state(source_path, size_bytes, modified_ns, content_sha256) VALUES (?, ?, ?, ?)",
                            (source_path, size_bytes, modified_ns, digest),
                        )
                        for start, end, chunk_text in rag._chunk_file(path):
                            chunk_digest = hashlib.sha256(
                                (source_path + "\0" + str(start) + "\0" + chunk_text).encode("utf-8")
                            ).hexdigest()
                            batch.append(
                                rag.RAGChunk(
                                    chunk_id=f"sha256:{chunk_digest}",
                                    source_path=source_path,
                                    text=chunk_text,
                                    start_line=start,
                                    end_line=end,
                                    sha256="sha256:" + hashlib.sha256(
                                        chunk_text.encode("utf-8")
                                    ).hexdigest(),
                                    metadata={},
                                )
                            )
                            if len(batch) >= batch_size:
                                flush_batch()
                    flush_batch()

                    connection.execute("DELETE FROM relations")
                    rag._insert_relations(connection, metadata)
                    rag._set_index_meta(connection, "metadata", rag._canonical_json(metadata))
                    rag._set_index_meta(connection, "files_indexed", str(len(current_stats)))
                    chunks_indexed = int(
                        connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                    )
                    rag._set_index_meta(connection, "chunks_indexed", str(chunks_indexed))
                    rag._set_index_meta(
                        connection,
                        "embedding_dimensions",
                        str(embedding_dimensions or 0),
                    )
                    if env_bool("MMM_RAG_INCREMENTAL_QUICK_CHECK", False):
                        quick = connection.execute("PRAGMA quick_check").fetchone()
                        if not quick or quick[0] != "ok":
                            raise ValueError("Incremental SQLite RAG update failed quick_check.")
                    connection.commit()
                    if semantic:
                        _ensure_semantic_lsh(perf, connection)
                except BaseException:
                    connection.rollback()
                    raise
                finally:
                    connection.close()
                    connection = None

                return {
                    "schema_version": "mmm/rag-build-result-v1",
                    "index_schema_version": self.schema_version,
                    "index_backend": "sqlite",
                    "lexical_backend": "sqlite_fts5" if fts5 else "deterministic_scan",
                    "index_path": str(target),
                    "files_indexed": len(current_stats),
                    "chunks_indexed": chunks_indexed,
                    "semantic_embeddings": semantic,
                    "embedding_dimensions": embedding_dimensions or 0,
                    "index_sha256": rag._sha256(target),
                    "incremental": True,
                    "changed_files": len(changed),
                    "removed_files": len(removed),
                    "metadata_only_files": len(metadata_only),
                    "reused_files": max(0, len(current_stats) - len(changed)),
                }
            except Exception:
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                return rebuild()

    setattr(build, "_mmm_research_rag_performance_v1", True)
    setattr(build, _MARKER, True)
    build.__wrapped__ = original  # type: ignore[attr-defined]
    return build


def harden(rag_index_module: Any, perf_module: Any) -> None:
    perf_module._bootstrap_incremental_state = _bootstrap_incremental_state

    def ensure(connection: sqlite3.Connection) -> None:
        _ensure_semantic_lsh(perf_module, connection)

    setattr(ensure, _MARKER, True)
    perf_module._ensure_semantic_lsh = ensure

    current = rag_index_module.ProjectRAGIndex.build
    if not getattr(current, _MARKER, False):
        rag_index_module.ProjectRAGIndex.build = _incremental_build_factory(
            rag_index_module,
            perf_module,
            current,
        )


__all__ = ["harden"]
