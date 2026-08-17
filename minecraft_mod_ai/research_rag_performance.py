from __future__ import annotations

"""Project-RAG hot path: content-addressed incremental updates and bounded ANN.

The dense path follows a multi-stage retrieval cascade: sparse/FTS candidates and
multi-probe random-hyperplane LSH generate a bounded set, then exact cosine and the
existing frozen reranker determine final order. Incremental indexing invalidates only
content-changed files; metadata-only mtime changes do not rechunk.
"""

import hashlib
import json
import math
import sqlite3
import threading
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .research_perf_common import env_int, table_exists

_MARKER = "_mmm_research_rag_performance_v1"
_RAG_LOCKS_GUARD = threading.RLock()
_RAG_LOCKS: dict[str, threading.RLock] = {}


def _path_lock(path: Path) -> threading.RLock:
    key = str(path.expanduser().resolve())
    with _RAG_LOCKS_GUARD:
        lock = _RAG_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _RAG_LOCKS[key] = lock
        return lock


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_rag_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    # source_commit is an expected per-repair mutation token. Relations are an
    # index-level graph and can be replaced without re-embedding/rechunking text.
    return {
        str(key): item
        for key, item in value.items()
        if str(key) not in {"source_commit", "relations"}
    }


def _initialize_incremental_state(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS mmm_file_state (
            source_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL,
            content_sha256 TEXT NOT NULL
        )
        """
    )


def _bootstrap_incremental_state(connection: sqlite3.Connection) -> None:
    count = int(connection.execute("SELECT COUNT(*) FROM mmm_file_state").fetchone()[0])
    if count:
        return
    rows = connection.execute(
        "SELECT source_path, size_bytes, modified_ns FROM indexed_files ORDER BY source_path"
    ).fetchall()
    for source_path, size_bytes, modified_ns in rows:
        path = Path(str(source_path))
        digest = _file_sha256(path) if path.is_file() and not path.is_symlink() else ""
        connection.execute(
            """
            INSERT OR REPLACE INTO mmm_file_state(
                source_path, size_bytes, modified_ns, content_sha256
            ) VALUES (?, ?, ?, ?)
            """,
            (str(source_path), int(size_bytes), int(modified_ns), digest),
        )


def _delete_rag_path(connection: sqlite3.Connection, source_path: str, *, fts5: bool) -> None:
    # Delete side indexes before the owning chunk rows so the subqueries still resolve.
    if table_exists(connection, "mmm_semantic_lsh"):
        connection.execute(
            """
            DELETE FROM mmm_semantic_lsh
            WHERE chunk_id IN (SELECT chunk_id FROM chunks WHERE source_path = ?)
            """,
            (source_path,),
        )
    if fts5 and table_exists(connection, "chunks_fts"):
        connection.execute(
            "DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE source_path = ?)",
            (source_path,),
        )
    connection.execute("DELETE FROM chunks WHERE source_path = ?", (source_path,))
    connection.execute("DELETE FROM indexed_files WHERE source_path = ?", (source_path,))
    connection.execute("DELETE FROM mmm_file_state WHERE source_path = ?", (source_path,))


def _incremental_rag_build_factory(rag: Any, original: Callable[..., dict[str, Any]]):
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
        if not target.is_file() or not rag._is_sqlite(target):
            result = original(
                self,
                roots,
                metadata=metadata,
                router=router,
                semantic=semantic,
                max_files=max_files,
            )
            try:
                if target.is_file() and rag._is_sqlite(target):
                    with sqlite3.connect(str(target)) as connection:
                        _initialize_incremental_state(connection)
                        _bootstrap_incremental_state(connection)
                        if semantic:
                            _ensure_semantic_lsh(connection)
                        connection.commit()
            except Exception:
                # Sidecar indexes are performance-only; the canonical build remains
                # authoritative if their initialization cannot complete safely.
                pass
            return result

        with _path_lock(target):
            try:
                connection = sqlite3.connect(str(target), timeout=30.0)
                connection.row_factory = sqlite3.Row
                meta = rag._read_index_meta(connection)
                if meta.get("schema_version") != self.schema_version:
                    connection.close()
                    return original(
                        self,
                        roots,
                        metadata=metadata,
                        router=router,
                        semantic=semantic,
                        max_files=max_files,
                    )
                semantic_existing = meta.get("semantic_embeddings") == "1"
                if semantic_existing != bool(semantic):
                    connection.close()
                    return original(
                        self,
                        roots,
                        metadata=metadata,
                        router=router,
                        semantic=semantic,
                        max_files=max_files,
                    )
                try:
                    previous_metadata = json.loads(meta.get("metadata", "{}"))
                except json.JSONDecodeError:
                    previous_metadata = {}
                if _stable_rag_metadata(previous_metadata) != _stable_rag_metadata(metadata):
                    connection.close()
                    return original(
                        self,
                        roots,
                        metadata=metadata,
                        router=router,
                        semantic=semantic,
                        max_files=max_files,
                    )

                current_paths = list(rag._iter_files(roots, max_files=max_files))
                current_stats: dict[str, tuple[int, int, Path]] = {}
                for path in current_paths:
                    stat = path.stat()
                    current_stats[str(path)] = (int(stat.st_size), int(stat.st_mtime_ns), path)

                _initialize_incremental_state(connection)
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
                    digest = _file_sha256(path)
                    if old is not None and old[2] == digest:
                        metadata_only.append((source_path, size_bytes, modified_ns, digest))
                    else:
                        changed.append((source_path, path, size_bytes, modified_ns, digest))

                fts5 = meta.get("fts5") == "1"
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for source_path in removed:
                        _delete_rag_path(connection, source_path, fts5=fts5)
                    for source_path, _path, _size, _mtime, _digest in changed:
                        _delete_rag_path(connection, source_path, fts5=fts5)
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
                                    sha256="sha256:" + hashlib.sha256(chunk_text.encode("utf-8")).hexdigest(),
                                    metadata={},
                                )
                            )
                            if len(batch) >= batch_size:
                                flush_batch()
                    flush_batch()

                    if previous_metadata.get("relations") != metadata.get("relations"):
                        connection.execute("DELETE FROM relations")
                        rag._insert_relations(connection, metadata)
                    rag._set_index_meta(connection, "metadata", rag._canonical_json(metadata))
                    rag._set_index_meta(connection, "files_indexed", str(len(current_stats)))
                    chunks_indexed = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
                    rag._set_index_meta(connection, "chunks_indexed", str(chunks_indexed))
                    rag._set_index_meta(
                        connection,
                        "embedding_dimensions",
                        str(embedding_dimensions or 0),
                    )
                    quick = connection.execute("PRAGMA quick_check").fetchone()
                    if not quick or quick[0] != "ok":
                        raise ValueError("Incremental SQLite RAG update failed quick_check.")
                    connection.commit()
                    if semantic:
                        _ensure_semantic_lsh(connection)
                except BaseException:
                    connection.rollback()
                    raise
                finally:
                    connection.close()

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
                # Correctness dominates optimization. If an old/corrupt index cannot be
                # upgraded safely, rebuild through the canonical implementation.
                try:
                    connection.close()  # type: ignore[possibly-undefined]
                except Exception:
                    pass
                return original(
                    self,
                    roots,
                    metadata=metadata,
                    router=router,
                    semantic=semantic,
                    max_files=max_files,
                )

    setattr(build, _MARKER, True)
    return build


# ---------------------------------------------------------------------------
# Multi-probe random-hyperplane LSH for bounded dense retrieval.
# ---------------------------------------------------------------------------

_LSH_BITS = 10
_LSH_SEEDS = (0x4D4D4D31, 0x4D4D4D32)


@lru_cache(maxsize=16)
def _projection_matrix(dimension: int, bits: int, seed: int):
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.standard_normal((bits, dimension), dtype=np.float32)


def _signatures(vectors: Sequence[Sequence[float]], *, bits: int = _LSH_BITS) -> list[tuple[int, int]]:
    if not vectors:
        return []
    import numpy as np

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] <= 0:
        return []
    result: list[list[int]] = [[], []]
    for table_index, seed in enumerate(_LSH_SEEDS):
        projection = _projection_matrix(int(array.shape[1]), bits, seed)
        signs = (array @ projection.T) >= 0.0
        weights = (1 << np.arange(bits, dtype=np.int64))
        packed = (signs.astype(np.int64) * weights).sum(axis=1)
        result[table_index] = [int(value) for value in packed.tolist()]
    return list(zip(result[0], result[1], strict=True))


def _parse_embedding(raw: Any) -> list[float]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    try:
        values = [float(item) for item in raw]
    except (TypeError, ValueError):
        return []
    return values if values and all(math.isfinite(item) for item in values) else []


def _ensure_semantic_lsh(connection: sqlite3.Connection) -> None:
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
            vector = _parse_embedding(row[1])
            if vector:
                valid_ids.append(str(row[0]))
                vectors.append(vector)
        if valid_ids:
            signatures = _signatures(vectors)
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
            # Mark invalid embeddings with an impossible bucket so migration progresses.
            connection.executemany(
                "INSERT OR REPLACE INTO mmm_semantic_lsh(chunk_id, sig_a, sig_b) VALUES (?, -1, -1)",
                [(chunk_id,) for chunk_id in invalid_ids],
            )
        connection.commit()


def _hamming_neighborhood(signature: int, bits: int, radius: int) -> list[int]:
    values = {int(signature)}
    if radius >= 1:
        values.update(signature ^ (1 << i) for i in range(bits))
    if radius >= 2:
        for i in range(bits):
            for j in range(i + 1, bits):
                values.add(signature ^ (1 << i) ^ (1 << j))
    return sorted(values)


def _lsh_candidate_rows(
    connection: sqlite3.Connection,
    query_vector: Sequence[float],
    *,
    target: int,
    cap: int,
) -> list[sqlite3.Row]:
    _ensure_semantic_lsh(connection)
    signatures = _signatures([query_vector])
    if not signatures:
        return []
    sig_a, sig_b = signatures[0]

    def query(radius: int) -> list[sqlite3.Row]:
        a = _hamming_neighborhood(sig_a, _LSH_BITS, radius)
        b = _hamming_neighborhood(sig_b, _LSH_BITS, radius)
        placeholders_a = ",".join("?" for _ in a)
        placeholders_b = ",".join("?" for _ in b)
        return connection.execute(
            f"""
            SELECT c.chunk_id, c.source_path, c.text, c.start_line, c.end_line,
                   c.sha256, c.embedding
            FROM mmm_semantic_lsh AS l
            JOIN chunks AS c ON c.chunk_id = l.chunk_id
            WHERE l.sig_a IN ({placeholders_a}) OR l.sig_b IN ({placeholders_b})
            ORDER BY c.source_path, c.start_line, c.chunk_id
            LIMIT ?
            """,
            (*a, *b, cap),
        ).fetchall()

    rows = query(1)
    if len(rows) < target:
        rows = query(2)
    return rows[:cap]


def _bounded_sqlite_search_pass_factory(rag: Any, original: Callable[..., Any]):
    @wraps(original)
    def bounded(
        connection: sqlite3.Connection,
        query: str,
        *,
        route: str,
        limit: int,
        metadata: dict[str, Any],
        router: Any | None,
        semantic: bool,
        rerank: bool,
        fts5_available: bool,
    ) -> Any:
        if not semantic:
            return original(
                connection,
                query,
                route=route,
                limit=limit,
                metadata=metadata,
                router=router,
                semantic=semantic,
                rerank=rerank,
                fts5_available=fts5_available,
            )
        assert router is not None
        exact_threshold = env_int(
            "MMM_RAG_EXACT_SCAN_THRESHOLD",
            4096,
            minimum=256,
            maximum=65536,
        )
        chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if chunk_count <= exact_threshold:
            return original(
                connection,
                query,
                route=route,
                limit=limit,
                metadata=metadata,
                router=router,
                semantic=semantic,
                rerank=rerank,
                fts5_available=fts5_available,
            )
        vectors = router.embed([query])
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("Embedding router did not return a query vector.")
        query_vector = vectors[0]
        candidate_limit = rag._candidate_budget(route, limit)
        cap = env_int(
            "MMM_RAG_DENSE_CANDIDATE_CAP",
            max(128, candidate_limit * 8),
            minimum=max(32, candidate_limit),
            maximum=2048,
        )
        query_terms = set(rag._meaningful_terms(query))
        query_lower = query.casefold()
        candidates: dict[str, Any] = {}
        lexical_backend = "deterministic_scan"

        if fts5_available:
            try:
                lexical_limit = min(cap, max(64, candidate_limit * 4))
                for row in rag._fts_rows(connection, query, lexical_limit):
                    chunk = rag._chunk_from_row(row, metadata)
                    lexical = rag._lexical_score(query_terms, query_lower, chunk.text)
                    semantic_score = rag._cosine(query_vector, chunk.embedding) if chunk.embedding else 0.0
                    if lexical > 0 or semantic_score > 0:
                        candidates[chunk.chunk_id] = rag._Candidate(
                            chunk=chunk,
                            lexical_score=lexical,
                            semantic_score=semantic_score,
                        )
                lexical_backend = "sqlite_fts5"
            except sqlite3.OperationalError:
                candidates.clear()
                lexical_backend = "deterministic_scan"

        try:
            dense_rows = _lsh_candidate_rows(
                connection,
                query_vector,
                target=max(candidate_limit * 2, 32),
                cap=cap,
            )
        except Exception:
            # No approximation is allowed to break retrieval correctness. Fall back
            # to the canonical exhaustive implementation if the side index fails.
            return original(
                connection,
                query,
                route=route,
                limit=limit,
                metadata=metadata,
                router=router,
                semantic=semantic,
                rerank=rerank,
                fts5_available=fts5_available,
            )

        minimum_dense = max(candidate_limit, limit * 2)
        if len(dense_rows) < minimum_dense:
            return original(
                connection,
                query,
                route=route,
                limit=limit,
                metadata=metadata,
                router=router,
                semantic=semantic,
                rerank=rerank,
                fts5_available=fts5_available,
            )

        for row in dense_rows:
            chunk = rag._chunk_from_row(row, metadata)
            lexical = rag._lexical_score(query_terms, query_lower, chunk.text)
            semantic_score = rag._cosine(query_vector, chunk.embedding) if chunk.embedding else 0.0
            candidate = rag._Candidate(
                chunk=chunk,
                lexical_score=lexical,
                semantic_score=semantic_score,
            )
            previous = candidates.get(chunk.chunk_id)
            if previous is None or candidate.score > previous.score:
                candidates[chunk.chunk_id] = candidate

        ordered = rag._top_candidates(list(candidates.values()), candidate_limit)
        if rerank and ordered:
            scores = router.rerank(query, [candidate.chunk.text for candidate in ordered])
            if len(scores) != len(ordered):
                raise ValueError("Reranker returned the wrong score count.")
            for candidate, score in zip(ordered, scores, strict=True):
                numeric_score = float(score)
                if not math.isfinite(numeric_score):
                    raise ValueError("Reranker returned a non-finite score.")
                candidate.reranker_score = numeric_score
            ordered = rag._top_candidates(ordered, candidate_limit)

        seed_limit = max(limit, min(16, candidate_limit))
        hits = [
            rag._candidate_to_hit(candidate)
            for candidate in ordered[:seed_limit]
            if candidate.score > 0
        ]
        relation_expansions = 0
        if route in {"multi_hop", "global_project"} and hits:
            related = rag._expand_sqlite_relationships(
                connection,
                query,
                hits,
                metadata=metadata,
                budget=max(limit, 8),
            )
            relation_expansions = len(related)
            hits = rag._merge_hits(hits, related)
        return rag._PassResult(
            hits=tuple(hits),
            candidates_considered=len(candidates),
            lexical_backend=(lexical_backend + "+lsh"),
            relation_expansions=relation_expansions,
        )

    setattr(bounded, _MARKER, True)
    return bounded


def _bounded_centroid_factory(centroid: Any, original: Callable[..., Any]):
    @wraps(original)
    def search(
        index_path: str | Path,
        *,
        query: str,
        q1_vector: Sequence[float],
        router: Any,
        limit: int = 8,
        required_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        target = Path(index_path).expanduser().resolve()
        if not target.is_file() or not centroid._is_sqlite(target) or not q1_vector:
            return None
        connection = sqlite3.connect(str(target))
        connection.row_factory = sqlite3.Row
        try:
            meta = {
                str(row[0]): str(row[1])
                for row in connection.execute("SELECT key, value FROM index_meta")
            }
            if meta.get("semantic_embeddings") != "1":
                return None
            try:
                index_metadata = json.loads(meta.get("metadata", "{}"))
            except json.JSONDecodeError:
                return None
            if not isinstance(index_metadata, Mapping) or not centroid._metadata_matches(
                index_metadata, required_metadata
            ):
                return None
            candidate_limit = max(limit * 6, 32)
            exact_threshold = env_int(
                "MMM_RAG_EXACT_SCAN_THRESHOLD",
                4096,
                minimum=256,
                maximum=65536,
            )
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
            if chunk_count <= exact_threshold:
                return original(
                    index_path,
                    query=query,
                    q1_vector=q1_vector,
                    router=router,
                    limit=limit,
                    required_metadata=required_metadata,
                )
            cap = env_int(
                "MMM_RAG_CENTROID_CANDIDATE_CAP",
                max(candidate_limit * 4, 128),
                minimum=candidate_limit,
                maximum=2048,
            )
            try:
                rows = _lsh_candidate_rows(
                    connection,
                    q1_vector,
                    target=candidate_limit,
                    cap=cap,
                )
            except Exception:
                return original(
                    index_path,
                    query=query,
                    q1_vector=q1_vector,
                    router=router,
                    limit=limit,
                    required_metadata=required_metadata,
                )
            if len(rows) < candidate_limit:
                return original(
                    index_path,
                    query=query,
                    q1_vector=q1_vector,
                    router=router,
                    limit=limit,
                    required_metadata=required_metadata,
                )
            candidates: list[dict[str, Any]] = []
            for row in rows:
                vector = centroid._vector(row["embedding"])
                if not vector:
                    continue
                score = centroid._cosine(q1_vector, vector)
                if not math.isfinite(score):
                    continue
                candidates.append(
                    {
                        "chunk_id": str(row["chunk_id"]),
                        "source_path": str(row["source_path"]),
                        "start_line": int(row["start_line"]),
                        "end_line": int(row["end_line"]),
                        "sha256": str(row["sha256"]),
                        "text": str(row["text"]),
                        "metadata": dict(index_metadata),
                        "lexical_score": 0.0,
                        "semantic_score": round(score, 6),
                        "reranker_score": 0.0,
                        "relation_score": 0.0,
                        "score": round(score, 6),
                    }
                )
            candidates.sort(
                key=lambda item: (-float(item["semantic_score"]), item["source_path"], item["start_line"])
            )
            candidates = candidates[:candidate_limit]
            if not candidates:
                return original(
                    index_path,
                    query=query,
                    q1_vector=q1_vector,
                    router=router,
                    limit=limit,
                    required_metadata=required_metadata,
                )
            try:
                reranked = router.rerank(query, [item["text"] for item in candidates])
            except Exception:
                reranked = []
            if len(reranked) == len(candidates):
                for hit, score in zip(candidates, reranked, strict=True):
                    numeric = float(score)
                    if math.isfinite(numeric):
                        hit["reranker_score"] = round(numeric, 6)
                        hit["score"] = round(float(hit["semantic_score"]) + 2.0 * numeric, 6)
                candidates.sort(
                    key=lambda item: (-float(item["score"]), item["source_path"], item["start_line"])
                )
            hits = candidates[:limit]
            terms = centroid._query_terms(query)
            covered = {
                term
                for term in terms
                if any(term in str(hit["text"]).casefold() for hit in hits)
            }
            coverage = len(covered) / max(1, len(terms)) if terms else 1.0
            relevance = max((float(hit["score"]) for hit in hits), default=0.0)
            return {
                "schema_version": "mmm/code-rag-result-v1",
                "query": query,
                "hits": hits,
                "receipt": {
                    "schema_version": "mmm/rag-search-receipt-v1",
                    "query": query,
                    "route": "centroid_vector",
                    "corrected_query": None,
                    "correction_applied": False,
                    "lexical_backend": "semantic_lsh",
                    "semantic_requested": True,
                    "semantic_used": True,
                    "rerank_requested": True,
                    "rerank_used": bool(reranked and len(reranked) == len(candidates)),
                    "candidates_considered": len(rows),
                    "relation_expansions": 0,
                    "result_count": len(hits),
                    "query_terms": sorted(terms),
                    "covered_terms": sorted(covered),
                    "missing_terms": sorted(terms - covered),
                    "coverage_score": round(coverage, 6),
                    "relevance_score": round(relevance, 6),
                    "required_metadata": dict(required_metadata or {}),
                    "warnings": [] if hits else ["no_relevant_chunks"],
                    "adaptation": "q0_topk_centroid_q1_lsh_exact_vector",
                },
                "retrieval_mode": "centroid-q1-lsh+exact-cosine+rerank",
                "centroid_adaptation": True,
                "centroid_vector_direct": True,
            }
        finally:
            connection.close()

    setattr(search, _MARKER, True)
    return search


def harden(rag_index_module: Any, centroid_module: Any) -> None:
    current_build = rag_index_module.ProjectRAGIndex.build
    if not getattr(current_build, _MARKER, False):
        rag_index_module.ProjectRAGIndex.build = _incremental_rag_build_factory(
            rag_index_module, current_build
        )

    current_search = rag_index_module._sqlite_search_pass
    if not getattr(current_search, _MARKER, False):
        rag_index_module._sqlite_search_pass = _bounded_sqlite_search_pass_factory(
            rag_index_module, current_search
        )

    current_centroid = centroid_module.direct_centroid_vector_search
    if not getattr(current_centroid, _MARKER, False):
        bounded_centroid = _bounded_centroid_factory(centroid_module, current_centroid)
        centroid_module.direct_centroid_vector_search = bounded_centroid
        try:
            from . import small_model_hybrid_search_contract as hybrid
            hybrid.direct_centroid_vector_search = bounded_centroid
        except Exception:
            pass


__all__ = ["harden"]
