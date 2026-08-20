from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .model_router import ModelRouter


_ALLOWED_SUFFIXES = frozenset(
    {
        ".java",
        ".json",
        ".gradle",
        ".kts",
        ".properties",
        ".md",
        ".txt",
        ".toml",
        ".yaml",
        ".yml",
        ".mcfunction",
        ".snbt",
    }
)
_TOKEN = re.compile(
    r"(?:[^\W\d_]|_)[\w.$:/-]*|\d+(?:\.\d+)+",
    flags=re.UNICODE,
)
_SQLITE_HEADER = b"SQLite format 3\x00"
_SQLITE_SCHEMA_VERSION = "mmm/project-rag-index-v2"
_LEGACY_SCHEMA_VERSION = "mmm/project-rag-index-v1"
_EMBEDDING_BATCH_SIZE = 64
_INSERT_BATCH_SIZE = 256
_MAX_TEXT_FRAGMENT_CHARACTERS = 8 * 1024
_MAX_CHUNK_CHARACTERS = 64 * 1024
_MAX_OVERLAP_CHARACTERS = 8 * 1024
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "what",
        "where",
        "which",
        "with",
        "all",
        "project",
        "please",
    }
)


@dataclass(frozen=True)
class RAGChunk:
    chunk_id: str
    source_path: str
    text: str
    start_line: int
    end_line: int
    sha256: str
    metadata: dict[str, Any]
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class RAGHit:
    chunk_id: str
    source_path: str
    start_line: int
    end_line: int
    score: float
    lexical_score: float
    semantic_score: float
    reranker_score: float
    text: str
    metadata: dict[str, Any]
    relation_score: float = 0.0


@dataclass(frozen=True)
class RAGSearchReceipt:
    """Auditable retrieval decisions, including weak-evidence outcomes."""

    schema_version: str
    query: str
    route: str
    corrected_query: str | None
    correction_applied: bool
    lexical_backend: str
    semantic_requested: bool
    semantic_used: bool
    rerank_requested: bool
    rerank_used: bool
    candidates_considered: int
    relation_expansions: int
    result_count: int
    query_terms: tuple[str, ...]
    covered_terms: tuple[str, ...]
    missing_terms: tuple[str, ...]
    coverage_score: float
    relevance_score: float
    required_metadata: dict[str, Any]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RAGSearchResult:
    hits: tuple[RAGHit, ...]
    receipt: RAGSearchReceipt


@dataclass
class _Candidate:
    chunk: RAGChunk
    lexical_score: float
    semantic_score: float
    reranker_score: float = 0.0
    relation_score: float = 0.0

    @property
    def score(self) -> float:
        return (
            self.lexical_score
            + self.semantic_score
            + (2.0 * self.reranker_score)
            + self.relation_score
        )


@dataclass(frozen=True)
class _PassResult:
    hits: tuple[RAGHit, ...]
    candidates_considered: int
    lexical_backend: str
    relation_expansions: int


class ProjectRAGIndex:
    """Durable, adaptive, version- and license-aware project retrieval.

    New indexes use SQLite even when an existing caller keeps a ``.json`` path.
    That preserves the public path contract while avoiding a single JSON payload
    that must be loaded into memory. Existing v1 JSON indexes remain readable.
    FTS5 is used when the host SQLite provides it; otherwise search uses a
    deterministic streaming lexical scan.
    """

    schema_version = _SQLITE_SCHEMA_VERSION

    def __init__(self, index_path: str | Path) -> None:
        self.index_path = Path(index_path).expanduser().resolve()

    def build(
        self,
        roots: Sequence[str | Path],
        *,
        metadata: dict[str, Any],
        router: ModelRouter | None = None,
        semantic: bool = False,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        """Build or incrementally refresh through the canonical RAG performance path."""
        from . import research_rag_performance

        return research_rag_performance.build_index(
            self,
            roots,
            metadata=metadata,
            router=router,
            semantic=semantic,
            max_files=max_files,
        )

    def _full_rebuild(
        self,
        roots: Sequence[str | Path],
        *,
        metadata: dict[str, Any],
        router: ModelRouter | None = None,
        semantic: bool = False,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        """Build an atomic SQLite index without a product-level file-count cap.

        ``max_files`` is retained as an opt-in host resource policy. The default
        is intentionally unbounded. Large source files are streamed into bounded
        fragments rather than silently disappearing from retrieval.
        """

        _validate_metadata(metadata)
        if semantic and router is None:
            raise ValueError("semantic=True requires a ModelRouter.")
        if max_files is not None and max_files < 1:
            raise ValueError("max_files must be positive when provided.")
        if self.index_path.exists() and self.index_path.is_dir():
            raise IsADirectoryError(self.index_path)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_name(
            f".{self.index_path.name}.{uuid.uuid4().hex}.tmp"
        )
        connection: sqlite3.Connection | None = None
        files_indexed = 0
        chunks_indexed = 0
        embedding_dimensions: int | None = None
        batch: list[RAGChunk] = []
        try:
            connection = sqlite3.connect(str(temporary_path))
            fts5_available = _initialize_sqlite(connection)
            metadata_json = _canonical_json(metadata)
            _set_index_meta(connection, "schema_version", self.schema_version)
            _set_index_meta(connection, "metadata", metadata_json)
            _set_index_meta(connection, "fts5", "1" if fts5_available else "0")
            _set_index_meta(
                connection,
                "semantic_embeddings",
                "1" if semantic else "0",
            )
            _insert_relations(connection, metadata)

            batch_size = (
                _EMBEDDING_BATCH_SIZE if semantic else _INSERT_BATCH_SIZE
            )
            for path in _iter_files(roots, max_files=max_files):
                stat = path.stat()
                inserted_file = connection.execute(
                    """
                    INSERT OR IGNORE INTO indexed_files(
                        source_path, size_bytes, modified_ns
                    ) VALUES (?, ?, ?)
                    """,
                    (str(path), int(stat.st_size), int(stat.st_mtime_ns)),
                ).rowcount
                if not inserted_file:
                    continue
                files_indexed += 1
                for start, end, chunk_text in _chunk_file(path):
                    digest = hashlib.sha256(
                        (
                            str(path)
                            + "\0"
                            + str(start)
                            + "\0"
                            + chunk_text
                        ).encode("utf-8")
                    ).hexdigest()
                    batch.append(
                        RAGChunk(
                            chunk_id=f"sha256:{digest}",
                            source_path=str(path),
                            text=chunk_text,
                            start_line=start,
                            end_line=end,
                            sha256=(
                                "sha256:"
                                + hashlib.sha256(
                                    chunk_text.encode("utf-8")
                                ).hexdigest()
                            ),
                            metadata={},
                        )
                    )
                    if len(batch) >= batch_size:
                        inserted, dimension = _insert_chunk_batch(
                            connection,
                            batch,
                            fts5_available=fts5_available,
                            router=router,
                            semantic=semantic,
                            expected_embedding_dimension=embedding_dimensions,
                        )
                        chunks_indexed += inserted
                        if dimension is not None:
                            embedding_dimensions = dimension
                        batch.clear()
                        connection.commit()

            if batch:
                inserted, dimension = _insert_chunk_batch(
                    connection,
                    batch,
                    fts5_available=fts5_available,
                    router=router,
                    semantic=semantic,
                    expected_embedding_dimension=embedding_dimensions,
                )
                chunks_indexed += inserted
                if dimension is not None:
                    embedding_dimensions = dimension
                batch.clear()

            _set_index_meta(connection, "files_indexed", str(files_indexed))
            _set_index_meta(connection, "chunks_indexed", str(chunks_indexed))
            _set_index_meta(
                connection,
                "embedding_dimensions",
                str(embedding_dimensions or 0),
            )
            connection.commit()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if not integrity or integrity[0] != "ok":
                raise ValueError("SQLite RAG index failed its integrity check.")
            connection.close()
            connection = None
            os.replace(temporary_path, self.index_path)
        except BaseException:
            if connection is not None:
                connection.close()
            temporary_path.unlink(missing_ok=True)
            raise

        return {
            # Keep the established result envelope while exposing the new
            # storage/schema details as additive fields.
            "schema_version": "mmm/rag-build-result-v1",
            "index_schema_version": self.schema_version,
            "index_backend": "sqlite",
            "lexical_backend": (
                "sqlite_fts5" if fts5_available else "deterministic_scan"
            ),
            "index_path": str(self.index_path),
            "files_indexed": files_indexed,
            "chunks_indexed": chunks_indexed,
            "semantic_embeddings": semantic,
            "embedding_dimensions": embedding_dimensions or 0,
            "index_sha256": _sha256(self.index_path),
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 8,
        router: ModelRouter | None = None,
        semantic: bool = False,
        rerank: bool = False,
        required_metadata: dict[str, Any] | None = None,
    ) -> list[RAGHit]:
        """Return hits using the adaptive route while preserving the v1 API."""

        return list(
            self.search_with_receipt(
                query,
                limit=limit,
                router=router,
                semantic=semantic,
                rerank=rerank,
                required_metadata=required_metadata,
            ).hits
        )

    def search_with_receipt(
        self,
        query: str,
        *,
        limit: int = 8,
        router: ModelRouter | None = None,
        semantic: bool = False,
        rerank: bool = False,
        required_metadata: dict[str, Any] | None = None,
    ) -> RAGSearchResult:
        """Search, evaluate evidence coverage, then correct at most once."""

        query = query.strip()
        if not query:
            raise ValueError("RAG query must not be empty.")
        if limit < 1:
            raise ValueError("RAG limit must be positive.")
        if (semantic or rerank) and router is None:
            requested = "semantic" if semantic else "rerank"
            raise ValueError(f"{requested}=True requires a ModelRouter.")
        if not self.index_path.is_file():
            raise FileNotFoundError(f"RAG index not found: {self.index_path}")

        route = _route_query(query)
        required = dict(required_metadata or {})
        if _is_sqlite(self.index_path):
            return self._search_sqlite(
                query,
                route=route,
                limit=limit,
                router=router,
                semantic=semantic,
                rerank=rerank,
                required_metadata=required,
            )
        return self._search_legacy(
            query,
            route=route,
            limit=limit,
            router=router,
            semantic=semantic,
            rerank=rerank,
            required_metadata=required,
        )

    def _search_sqlite(
        self,
        query: str,
        *,
        route: str,
        limit: int,
        router: ModelRouter | None,
        semantic: bool,
        rerank: bool,
        required_metadata: dict[str, Any],
    ) -> RAGSearchResult:
        connection = sqlite3.connect(str(self.index_path))
        connection.row_factory = sqlite3.Row
        try:
            index_meta = _read_index_meta(connection)
            if index_meta.get("schema_version") != self.schema_version:
                raise ValueError("Unsupported SQLite RAG index schema.")
            metadata = json.loads(index_meta.get("metadata", "{}"))
            _validate_metadata(metadata)
            fts5_available = index_meta.get("fts5") == "1"
            semantic_available = (
                index_meta.get("semantic_embeddings") == "1"
            )
            if semantic and not semantic_available:
                raise ValueError(
                    "semantic=True was requested, but this index has no "
                    "semantic embeddings."
                )
            if not _metadata_matches(metadata, required_metadata):
                return _empty_result(
                    query,
                    route=route,
                    semantic=semantic,
                    rerank=rerank,
                    required_metadata=required_metadata,
                    lexical_backend=(
                        "sqlite_fts5"
                        if fts5_available
                        else "deterministic_scan"
                    ),
                    warning="required_metadata_mismatch",
                )

            first = _sqlite_search_pass(
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
            first_coverage = _coverage(query, first.hits)
            threshold = _coverage_threshold(route)
            corrected_query: str | None = None
            correction_applied = False
            passes = [first]
            hits = list(first.hits)
            if first_coverage[0] < threshold:
                corrected_query = _correct_query(
                    query,
                    metadata=metadata,
                    missing_terms=first_coverage[2],
                )
                if corrected_query.casefold() != query.casefold():
                    correction_applied = True
                    second = _sqlite_search_pass(
                        connection,
                        corrected_query,
                        route=route,
                        limit=limit,
                        metadata=metadata,
                        router=router,
                        semantic=semantic,
                        rerank=rerank,
                        fts5_available=fts5_available,
                    )
                    passes.append(second)
                    hits = _merge_hits(hits, second.hits)

            hits = _finalize_hits(hits, route=route, limit=limit)
            coverage, covered, missing = _coverage(query, hits)
            warnings: list[str] = []
            if not hits:
                warnings.append("no_relevant_chunks")
            if coverage < threshold:
                warnings.append("coverage_below_route_threshold")
            return _result_with_receipt(
                query,
                route=route,
                hits=hits,
                corrected_query=corrected_query,
                correction_applied=correction_applied,
                lexical_backend=passes[0].lexical_backend,
                semantic=semantic,
                rerank=rerank,
                candidates_considered=sum(
                    item.candidates_considered for item in passes
                ),
                relation_expansions=sum(
                    item.relation_expansions for item in passes
                ),
                covered=covered,
                missing=missing,
                coverage=coverage,
                required_metadata=required_metadata,
                warnings=warnings,
            )
        finally:
            connection.close()

    def _search_legacy(
        self,
        query: str,
        *,
        route: str,
        limit: int,
        router: ModelRouter | None,
        semantic: bool,
        rerank: bool,
        required_metadata: dict[str, Any],
    ) -> RAGSearchResult:
        chunks = [
            chunk
            for chunk in self._load_legacy()
            if _metadata_matches(chunk.metadata, required_metadata)
        ]
        if chunks:
            _validate_metadata(chunks[0].metadata)
        if semantic and not any(chunk.embedding for chunk in chunks):
            raise ValueError(
                "semantic=True was requested, but this legacy index has no "
                "semantic embeddings."
            )
        if not chunks:
            return _empty_result(
                query,
                route=route,
                semantic=semantic,
                rerank=rerank,
                required_metadata=required_metadata,
                lexical_backend="legacy_scan",
                warning=(
                    "required_metadata_mismatch"
                    if required_metadata
                    else "no_relevant_chunks"
                ),
            )

        first = _legacy_search_pass(
            chunks,
            query,
            route=route,
            limit=limit,
            router=router,
            semantic=semantic,
            rerank=rerank,
        )
        coverage, _, missing = _coverage(query, first.hits)
        threshold = _coverage_threshold(route)
        corrected_query: str | None = None
        correction_applied = False
        passes = [first]
        hits = list(first.hits)
        if coverage < threshold:
            corrected_query = _correct_query(
                query,
                metadata=chunks[0].metadata,
                missing_terms=missing,
            )
            if corrected_query.casefold() != query.casefold():
                correction_applied = True
                second = _legacy_search_pass(
                    chunks,
                    corrected_query,
                    route=route,
                    limit=limit,
                    router=router,
                    semantic=semantic,
                    rerank=rerank,
                )
                passes.append(second)
                hits = _merge_hits(hits, second.hits)

        hits = _finalize_hits(hits, route=route, limit=limit)
        coverage, covered, missing = _coverage(query, hits)
        warnings: list[str] = []
        if not hits:
            warnings.append("no_relevant_chunks")
        if coverage < threshold:
            warnings.append("coverage_below_route_threshold")
        return _result_with_receipt(
            query,
            route=route,
            hits=hits,
            corrected_query=corrected_query,
            correction_applied=correction_applied,
            lexical_backend="legacy_scan",
            semantic=semantic,
            rerank=rerank,
            candidates_considered=sum(
                item.candidates_considered for item in passes
            ),
            relation_expansions=sum(
                item.relation_expansions for item in passes
            ),
            covered=covered,
            missing=missing,
            coverage=coverage,
            required_metadata=required_metadata,
            warnings=warnings,
        )

    def _load(self) -> list[RAGChunk]:
        """Compatibility helper for callers that used the former private API."""

        if not self.index_path.is_file():
            raise FileNotFoundError(f"RAG index not found: {self.index_path}")
        if not _is_sqlite(self.index_path):
            return self._load_legacy()
        connection = sqlite3.connect(str(self.index_path))
        connection.row_factory = sqlite3.Row
        try:
            index_meta = _read_index_meta(connection)
            metadata = json.loads(index_meta.get("metadata", "{}"))
            return [
                _chunk_from_row(row, metadata)
                for row in connection.execute(
                    """
                    SELECT chunk_id, source_path, text, start_line, end_line,
                           sha256, embedding
                    FROM chunks
                    ORDER BY source_path, start_line, chunk_id
                    """
                )
            ]
        finally:
            connection.close()

    def _load_legacy(self) -> list[RAGChunk]:
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != _LEGACY_SCHEMA_VERSION:
            raise ValueError("Unsupported RAG index schema.")
        result: list[RAGChunk] = []
        for item in raw.get("chunks", []):
            result.append(
                RAGChunk(
                    chunk_id=item["chunk_id"],
                    source_path=item["source_path"],
                    text=item["text"],
                    start_line=int(item["start_line"]),
                    end_line=int(item["end_line"]),
                    sha256=item["sha256"],
                    metadata=dict(item["metadata"]),
                    embedding=tuple(
                        float(value) for value in item.get("embedding", [])
                    ),
                )
            )
        return result


def _initialize_sqlite(connection: sqlite3.Connection) -> bool:
    connection.executescript(
        """
        PRAGMA journal_mode = DELETE;
        PRAGMA synchronous = NORMAL;
        PRAGMA temp_store = MEMORY;
        CREATE TABLE index_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE indexed_files (
            source_path TEXT PRIMARY KEY,
            size_bytes INTEGER NOT NULL,
            modified_ns INTEGER NOT NULL
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT NOT NULL UNIQUE,
            source_path TEXT NOT NULL,
            normalized_path TEXT NOT NULL,
            text TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            embedding TEXT NOT NULL DEFAULT '[]'
        );
        CREATE INDEX chunks_source_location
            ON chunks(source_path, start_line, chunk_id);
        CREATE INDEX chunks_normalized_path
            ON chunks(normalized_path, start_line);
        CREATE TABLE relations (
            source TEXT NOT NULL,
            target TEXT NOT NULL,
            kind TEXT NOT NULL,
            PRIMARY KEY(source, target, kind)
        );
        CREATE INDEX relations_source ON relations(source);
        """
    )
    try:
        connection.execute(
            """
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                source_path,
                text,
                tokenize = 'unicode61'
            )
            """
        )
    except sqlite3.OperationalError:
        return False
    return True


def _set_index_meta(
    connection: sqlite3.Connection,
    key: str,
    value: str,
) -> None:
    connection.execute(
        """
        INSERT INTO index_meta(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _read_index_meta(connection: sqlite3.Connection) -> dict[str, str]:
    try:
        return {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT key, value FROM index_meta ORDER BY key"
            )
        }
    except sqlite3.DatabaseError as exc:
        raise ValueError("Invalid SQLite RAG index.") from exc


def _insert_chunk_batch(
    connection: sqlite3.Connection,
    chunks: Sequence[RAGChunk],
    *,
    fts5_available: bool,
    router: ModelRouter | None,
    semantic: bool,
    expected_embedding_dimension: int | None,
) -> tuple[int, int | None]:
    if semantic:
        assert router is not None
        vectors = router.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("Embedding router returned the wrong vector count.")
    else:
        vectors = [[] for _ in chunks]

    inserted = 0
    dimension = expected_embedding_dimension
    for chunk, raw_vector in zip(chunks, vectors, strict=True):
        vector = tuple(float(value) for value in raw_vector)
        if any(not math.isfinite(value) for value in vector):
            raise ValueError("Embedding router returned a non-finite value.")
        if semantic and not vector:
            raise ValueError("Embedding router returned an empty vector.")
        if vector:
            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(
                    "Embedding router returned inconsistent vector dimensions."
                )
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO chunks(
                chunk_id, source_path, normalized_path, text,
                start_line, end_line, sha256, embedding
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.chunk_id,
                chunk.source_path,
                _normalize_relation_path(chunk.source_path),
                chunk.text,
                chunk.start_line,
                chunk.end_line,
                chunk.sha256,
                _canonical_json(vector),
            ),
        )
        if not cursor.rowcount:
            continue
        inserted += 1
        if fts5_available:
            connection.execute(
                """
                INSERT INTO chunks_fts(rowid, chunk_id, source_path, text)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cursor.lastrowid,
                    chunk.chunk_id,
                    chunk.source_path,
                    chunk.text,
                ),
            )
    return inserted, dimension


def _insert_relations(
    connection: sqlite3.Connection,
    metadata: dict[str, Any],
) -> None:
    connection.executemany(
        """
        INSERT OR IGNORE INTO relations(source, target, kind)
        VALUES (?, ?, ?)
        """,
        _extract_relations(metadata),
    )


def _sqlite_search_pass(
    connection: sqlite3.Connection,
    query: str,
    *,
    route: str,
    limit: int,
    metadata: dict[str, Any],
    router: ModelRouter | None,
    semantic: bool,
    rerank: bool,
    fts5_available: bool,
) -> _PassResult:
    """Use bounded ANN when eligible, with the exhaustive pass as correctness fallback."""
    from . import research_rag_performance

    return research_rag_performance.sqlite_search_pass(
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


def _full_sqlite_search_pass(
    connection: sqlite3.Connection,
    query: str,
    *,
    route: str,
    limit: int,
    metadata: dict[str, Any],
    router: ModelRouter | None,
    semantic: bool,
    rerank: bool,
    fts5_available: bool,
) -> _PassResult:
    candidate_limit = _candidate_budget(route, limit)
    query_terms = set(_meaningful_terms(query))
    query_lower = query.casefold()
    query_vector: Sequence[float] | None = None
    if semantic:
        assert router is not None
        vectors = router.embed([query])
        if len(vectors) != 1 or not vectors[0]:
            raise ValueError("Embedding router did not return a query vector.")
        query_vector = vectors[0]

    candidates: dict[str, _Candidate] = {}
    lexical_backend = "deterministic_scan"
    if fts5_available:
        try:
            for row in _fts_rows(connection, query, candidate_limit):
                chunk = _chunk_from_row(row, metadata)
                lexical = _lexical_score(
                    query_terms,
                    query_lower,
                    chunk.text,
                )
                if lexical > 0:
                    candidates[chunk.chunk_id] = _Candidate(
                        chunk=chunk,
                        lexical_score=lexical,
                        semantic_score=0.0,
                    )
            lexical_backend = "sqlite_fts5"
        except sqlite3.OperationalError:
            # FTS syntax/build differences must not make lexical retrieval
            # unavailable. The fallback remains deterministic and bounded.
            candidates.clear()
            lexical_backend = "deterministic_scan"

    if semantic or lexical_backend == "deterministic_scan" or not candidates:
        streamed: list[_Candidate] = []
        for row in connection.execute(
            """
            SELECT chunk_id, source_path, text, start_line, end_line,
                   sha256, embedding
            FROM chunks
            ORDER BY source_path, start_line, chunk_id
            """
        ):
            chunk = _chunk_from_row(row, metadata)
            lexical = _lexical_score(query_terms, query_lower, chunk.text)
            semantic_score = (
                _cosine(query_vector, chunk.embedding)
                if query_vector is not None and chunk.embedding
                else 0.0
            )
            candidate = _Candidate(
                chunk=chunk,
                lexical_score=lexical,
                semantic_score=semantic_score,
            )
            if candidate.score > 0:
                streamed.append(candidate)
            if len(streamed) >= candidate_limit * 2:
                streamed = _top_candidates(streamed, candidate_limit)
        for candidate in _top_candidates(streamed, candidate_limit):
            previous = candidates.get(candidate.chunk.chunk_id)
            if previous is None or candidate.score > previous.score:
                candidates[candidate.chunk.chunk_id] = candidate

    ordered = _top_candidates(list(candidates.values()), candidate_limit)
    if rerank and ordered:
        assert router is not None
        scores = router.rerank(
            query,
            [candidate.chunk.text for candidate in ordered],
        )
        if len(scores) != len(ordered):
            raise ValueError("Reranker returned the wrong score count.")
        for candidate, score in zip(ordered, scores, strict=True):
            numeric_score = float(score)
            if not math.isfinite(numeric_score):
                raise ValueError("Reranker returned a non-finite score.")
            candidate.reranker_score = numeric_score
        ordered = _top_candidates(ordered, candidate_limit)

    seed_limit = max(limit, min(16, candidate_limit))
    hits = [
        _candidate_to_hit(candidate)
        for candidate in ordered[:seed_limit]
        if candidate.score > 0
    ]
    relation_expansions = 0
    if route in {"multi_hop", "global_project"} and hits:
        related = _expand_sqlite_relationships(
            connection,
            query,
            hits,
            metadata=metadata,
            budget=max(limit, 8),
        )
        relation_expansions = len(related)
        hits = _merge_hits(hits, related)
    return _PassResult(
        hits=tuple(hits),
        candidates_considered=len(candidates),
        lexical_backend=lexical_backend,
        relation_expansions=relation_expansions,
    )


def _legacy_search_pass(
    chunks: Sequence[RAGChunk],
    query: str,
    *,
    route: str,
    limit: int,
    router: ModelRouter | None,
    semantic: bool,
    rerank: bool,
) -> _PassResult:
    query_terms = set(_meaningful_terms(query))
    query_lower = query.casefold()
    query_vector: Sequence[float] | None = None
    if semantic:
        assert router is not None
        query_vector = router.embed([query])[0]
    budget = _candidate_budget(route, limit)
    candidates: list[_Candidate] = []
    for chunk in chunks:
        lexical = _lexical_score(query_terms, query_lower, chunk.text)
        semantic_score = (
            _cosine(query_vector, chunk.embedding)
            if query_vector is not None and chunk.embedding
            else 0.0
        )
        candidate = _Candidate(
            chunk=chunk,
            lexical_score=lexical,
            semantic_score=semantic_score,
        )
        if candidate.score > 0:
            candidates.append(candidate)
        if len(candidates) >= budget * 2:
            candidates = _top_candidates(candidates, budget)
    candidates = _top_candidates(candidates, budget)
    if rerank and candidates:
        assert router is not None
        scores = router.rerank(
            query,
            [candidate.chunk.text for candidate in candidates],
        )
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned the wrong score count.")
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.reranker_score = float(score)
        candidates = _top_candidates(candidates, budget)
    hits = [
        _candidate_to_hit(candidate)
        for candidate in candidates[: max(limit, min(16, budget))]
    ]
    related: list[RAGHit] = []
    if route in {"multi_hop", "global_project"} and hits:
        related = _expand_legacy_relationships(
            chunks,
            query,
            hits,
            budget=max(limit, 8),
        )
        hits = _merge_hits(hits, related)
    return _PassResult(
        hits=tuple(hits),
        candidates_considered=len(candidates),
        lexical_backend="legacy_scan",
        relation_expansions=len(related),
    )


def _fts_rows(
    connection: sqlite3.Connection,
    query: str,
    limit: int,
) -> Iterator[sqlite3.Row]:
    terms = _meaningful_terms(query)
    if not terms:
        return
    fts_query = " OR ".join(
        '"' + term.replace('"', '""') + '"' for term in terms
    )
    yield from connection.execute(
        """
        SELECT c.chunk_id, c.source_path, c.text, c.start_line, c.end_line,
               c.sha256, c.embedding, bm25(chunks_fts) AS fts_rank
        FROM chunks_fts
        JOIN chunks AS c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY fts_rank ASC, c.source_path, c.start_line, c.chunk_id
        LIMIT ?
        """,
        (fts_query, limit),
    )


def _expand_sqlite_relationships(
    connection: sqlite3.Connection,
    query: str,
    seeds: Sequence[RAGHit],
    *,
    metadata: dict[str, Any],
    budget: int,
) -> list[RAGHit]:
    existing = {hit.chunk_id for hit in seeds}
    query_terms = set(_meaningful_terms(query))
    query_lower = query.casefold()
    related: list[RAGHit] = []
    seen_edges: set[tuple[str, str, str]] = set()
    for seed in seeds[:16]:
        aliases = _path_aliases(seed.source_path)
        if not aliases:
            continue
        placeholders = ",".join("?" for _ in aliases)
        rows = connection.execute(
            f"""
            SELECT source, target, kind
            FROM relations
            WHERE source IN ({placeholders})
            ORDER BY source, target, kind
            """,
            tuple(aliases),
        )
        for relation in rows:
            edge = (
                str(relation["source"]),
                str(relation["target"]),
                str(relation["kind"]),
            )
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            target = edge[1]
            escaped_target = _escape_like(target)
            target_rows = connection.execute(
                """
                SELECT chunk_id, source_path, text, start_line, end_line,
                       sha256, embedding
                FROM chunks
                WHERE normalized_path = ?
                   OR normalized_path LIKE ? ESCAPE '!'
                ORDER BY source_path, start_line, chunk_id
                LIMIT 2
                """,
                (target, f"%/{escaped_target}"),
            )
            for row in target_rows:
                chunk = _chunk_from_row(row, metadata)
                if chunk.chunk_id in existing:
                    continue
                lexical = _lexical_score(
                    query_terms,
                    query_lower,
                    chunk.text,
                )
                relation_score = max(0.1, min(1.0, seed.score * 0.15))
                relation_metadata = dict(metadata)
                relation_metadata["_rag_relation"] = {
                    "source": seed.source_path,
                    "target": chunk.source_path,
                    "kind": edge[2],
                }
                related.append(
                    RAGHit(
                        chunk_id=chunk.chunk_id,
                        source_path=chunk.source_path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        score=round(lexical + relation_score, 6),
                        lexical_score=round(lexical, 6),
                        semantic_score=0.0,
                        reranker_score=0.0,
                        text=chunk.text,
                        metadata=relation_metadata,
                        relation_score=round(relation_score, 6),
                    )
                )
                existing.add(chunk.chunk_id)
                if len(related) >= budget:
                    return related
    return related


def _expand_legacy_relationships(
    chunks: Sequence[RAGChunk],
    query: str,
    seeds: Sequence[RAGHit],
    *,
    budget: int,
) -> list[RAGHit]:
    relations = _extract_relations(seeds[0].metadata if seeds else {})
    if not relations:
        return []
    by_alias: dict[str, list[RAGChunk]] = {}
    for chunk in chunks:
        for alias in _path_aliases(chunk.source_path):
            by_alias.setdefault(alias, []).append(chunk)
    existing = {hit.chunk_id for hit in seeds}
    query_terms = set(_meaningful_terms(query))
    query_lower = query.casefold()
    result: list[RAGHit] = []
    for seed in seeds[:16]:
        aliases = set(_path_aliases(seed.source_path))
        for source, target, kind in relations:
            if source not in aliases:
                continue
            for chunk in by_alias.get(target, [])[:2]:
                if chunk.chunk_id in existing:
                    continue
                lexical = _lexical_score(
                    query_terms,
                    query_lower,
                    chunk.text,
                )
                relation_score = max(0.1, min(1.0, seed.score * 0.15))
                metadata = dict(chunk.metadata)
                metadata["_rag_relation"] = {
                    "source": seed.source_path,
                    "target": chunk.source_path,
                    "kind": kind,
                }
                result.append(
                    RAGHit(
                        chunk_id=chunk.chunk_id,
                        source_path=chunk.source_path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        score=round(lexical + relation_score, 6),
                        lexical_score=round(lexical, 6),
                        semantic_score=0.0,
                        reranker_score=0.0,
                        text=chunk.text,
                        metadata=metadata,
                        relation_score=round(relation_score, 6),
                    )
                )
                existing.add(chunk.chunk_id)
                if len(result) >= budget:
                    return result
    return result


def _extract_relations(
    metadata: dict[str, Any],
) -> list[tuple[str, str, str]]:
    raw = metadata.get("relations")
    relations: list[tuple[str, str, str]] = []
    if isinstance(raw, dict):
        for source, targets in raw.items():
            if isinstance(targets, str):
                target_values: Sequence[Any] = [targets]
            elif isinstance(targets, dict):
                target_values = targets.get("targets", [])
            elif isinstance(targets, Sequence):
                target_values = targets
            else:
                continue
            for target in target_values:
                if isinstance(target, dict):
                    target_value = target.get("target")
                    kind = str(target.get("kind", "depends_on"))
                else:
                    target_value = target
                    kind = "depends_on"
                if target_value:
                    relations.append(
                        (
                            _normalize_relation_path(str(source)),
                            _normalize_relation_path(str(target_value)),
                            kind,
                        )
                    )
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            targets = item.get("targets", item.get("target"))
            if isinstance(targets, str):
                target_values = [targets]
            elif isinstance(targets, Sequence):
                target_values = list(targets)
            else:
                target_values = []
            for target in target_values:
                if source and target:
                    relations.append(
                        (
                            _normalize_relation_path(str(source)),
                            _normalize_relation_path(str(target)),
                            str(item.get("kind", "depends_on")),
                        )
                    )
    return sorted(set(relations))


def _validate_metadata(metadata: dict[str, Any]) -> None:
    required = {
        "minecraft_version",
        "loader",
        "mapping_namespace",
        "java_version",
        "license",
        "source_commit",
    }
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"RAG metadata is missing: {sorted(missing)}")
    for field in ("minecraft_version", "loader", "java_version"):
        if not str(metadata[field]).strip():
            raise ValueError(f"RAG metadata field {field} must be non-empty.")
    if metadata["mapping_namespace"] not in {
        "yarn",
        "intermediary",
        "official",
    }:
        raise ValueError("Unsupported mapping namespace.")
    if (
        not str(metadata["license"]).strip()
        or not str(metadata["source_commit"]).strip()
    ):
        raise ValueError("RAG source license and commit are required.")


def _metadata_matches(
    metadata: dict[str, Any],
    required: dict[str, Any],
) -> bool:
    return all(metadata.get(key) == value for key, value in required.items())


def _iter_files(
    roots: Sequence[str | Path],
    *,
    max_files: int | None = None,
) -> Iterable[Path]:
    """Yield files in stable order without materializing the project tree."""

    found = 0
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(root)
        if root.is_file():
            candidates: Iterable[Path] = (root,)
        else:
            candidates = _walk_files(root)
        for path in candidates:
            if (
                path.is_file()
                and not path.is_symlink()
                and path.suffix.lower() in _ALLOWED_SUFFIXES
            ):
                if max_files is not None and found >= max_files:
                    raise ValueError(f"RAG file limit exceeded: {max_files}")
                found += 1
                yield path


def _walk_files(root: Path) -> Iterator[Path]:
    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        directory_names[:] = sorted(
            name
            for name in directory_names
            if not (Path(directory) / name).is_symlink()
        )
        for file_name in sorted(file_names):
            yield Path(directory) / file_name


def _chunk_file(path: Path) -> Iterable[tuple[int, int, str]]:
    if path.suffix.lower() in {
        ".java",
        ".gradle",
        ".kts",
        ".mcfunction",
        ".snbt",
    }:
        size, overlap = 160, 24
    else:
        size, overlap = 100, 15

    # ``readline(size)`` bounds memory even for minified JSON or generated source
    # containing a multi-megabyte physical line. A fragment that does not end in a
    # newline retains the same source-line number as the following fragment.
    lines: list[tuple[int, str]] = []
    buffered_characters = 0
    start_line = 1
    final_line = 0
    fragments_seen = 0
    last_emitted_fragment = 0
    with path.open("r", encoding="utf-8", errors="replace") as source:
        line_number = 1
        while True:
            line = source.readline(_MAX_TEXT_FRAGMENT_CHARACTERS)
            if not line:
                break
            fragment_line = line_number
            final_line = fragment_line
            fragments_seen += 1
            fragment = line.rstrip("\r\n")
            lines.append((fragment_line, fragment))
            buffered_characters += len(fragment) + 1
            if line.endswith(("\n", "\r")):
                line_number += 1
            if (
                len(lines) < size
                and buffered_characters < _MAX_CHUNK_CHARACTERS
            ):
                continue
            chunk = "\n".join(value for _, value in lines).strip()
            if chunk:
                yield start_line, fragment_line, chunk
            last_emitted_fragment = fragments_seen
            kept: list[tuple[int, str]] = []
            kept_characters = 0
            for entry in reversed(lines[-overlap:]):
                entry_size = len(entry[1]) + 1
                if (
                    kept
                    and kept_characters + entry_size
                    > _MAX_OVERLAP_CHARACTERS
                ):
                    break
                kept.append(entry)
                kept_characters += entry_size
            lines = list(reversed(kept))
            buffered_characters = kept_characters
            start_line = lines[0][0] if lines else line_number
    if lines and fragments_seen > last_emitted_fragment:
        chunk = "\n".join(value for _, value in lines).strip()
        if chunk:
            yield start_line, final_line, chunk


def _chunk_text(path: Path, text: str) -> Iterable[tuple[int, int, str]]:
    """Retained for compatibility with focused callers and tests."""

    lines = text.splitlines()
    if path.suffix.lower() in {
        ".java",
        ".gradle",
        ".kts",
        ".mcfunction",
        ".snbt",
    }:
        size, overlap = 160, 24
    else:
        size, overlap = 100, 15
    start = 0
    while start < len(lines):
        end = min(len(lines), start + size)
        chunk = "\n".join(lines[start:end]).strip()
        if chunk:
            yield start + 1, end, chunk
        if end >= len(lines):
            break
        start = max(start + 1, end - overlap)


def _route_query(query: str) -> str:
    lowered = query.casefold()
    terms = _meaningful_terms(query)
    if re.search(r"\b\d+\.\d+(?:\.\d+)?\b", query) or any(
        marker in lowered
        for marker in (
            "yarn",
            "mapping",
            "signature",
            "fabric api version",
            "minecraft version",
        )
    ):
        return "exact_version"
    if any(
        marker in lowered
        for marker in (
            "architecture",
            "entire project",
            "whole project",
            "global",
            "overview",
            "across the project",
            "프로젝트 전체",
            "전체 구조",
        )
    ):
        return "global_project"
    if (
        any(
            marker in lowered
            for marker in (
                "depends",
                "dependency",
                "call chain",
                "relationship",
                "related",
                "affect",
                "flow from",
                "의존",
                "연결",
                "관계",
            )
        )
        or (" and " in lowered and len(terms) >= 7)
        or len(terms) >= 12
    ):
        return "multi_hop"
    return "single"


def _candidate_budget(route: str, limit: int) -> int:
    multiplier = {
        "exact_version": 16,
        "single": 12,
        "multi_hop": 24,
        "global_project": 32,
    }[route]
    floor = {
        "exact_version": 128,
        "single": 96,
        "multi_hop": 256,
        "global_project": 512,
    }[route]
    return max(limit * multiplier, floor)


def _coverage_threshold(route: str) -> float:
    return {
        "exact_version": 0.60,
        "single": 0.55,
        "multi_hop": 0.65,
        "global_project": 0.50,
    }[route]


def _correct_query(
    query: str,
    *,
    metadata: dict[str, Any],
    missing_terms: Sequence[str],
) -> str:
    aliases = {
        "datagen": ("data", "generator"),
        "register": ("registry",),
        "registration": ("registry",),
        "biomes": ("biome", "worldgen"),
        "dimensions": ("dimension", "worldgen"),
        "mixins": ("mixin", "injection"),
        "recipes": ("recipe", "json"),
        "tags": ("tag", "json"),
    }
    additions: list[str] = []
    for term in missing_terms:
        additions.extend(aliases.get(term, ()))
        additions.extend(_split_identifier(term))
    additions.extend(
        [
            str(metadata.get("loader", "")),
            str(metadata.get("minecraft_version", "")),
            str(metadata.get("mapping_namespace", "")),
        ]
    )
    existing = set(_tokens(query))
    unique_additions: list[str] = []
    for addition in additions:
        normalized = addition.strip()
        if normalized and normalized.casefold() not in existing:
            existing.add(normalized.casefold())
            unique_additions.append(normalized)
    if not unique_additions:
        return query
    return query + " " + " ".join(unique_additions)


def _split_identifier(value: str) -> list[str]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)
    return [
        part.casefold()
        for part in re.split(r"[_.$:/-]+|\s+", separated)
        if len(part) >= 3
    ]


def _coverage(
    query: str,
    hits: Sequence[RAGHit],
) -> tuple[float, tuple[str, ...], tuple[str, ...]]:
    query_terms = _meaningful_terms(query)
    if not query_terms:
        return 1.0, (), ()
    evidence_terms: set[str] = set()
    for hit in hits:
        evidence_terms.update(_tokens(hit.text))
        evidence_terms.update(_tokens(hit.source_path))
        evidence_terms.update(_tokens(_canonical_json(hit.metadata)))
    covered = tuple(term for term in query_terms if term in evidence_terms)
    missing = tuple(term for term in query_terms if term not in evidence_terms)
    return round(len(covered) / len(query_terms), 6), covered, missing


def _lexical_score(
    query_terms: set[str],
    query_lower: str,
    text: str,
) -> float:
    if not query_terms:
        return 0.0
    chunk_terms = set(_tokens(text))
    intersection = len(query_terms & chunk_terms)
    text_lower = text.casefold()
    exact_bonus = sum(
        2.0 for term in query_terms if term and term in text_lower
    )
    phrase_bonus = 1.0 if len(query_lower) >= 4 and query_lower in text_lower else 0.0
    return (intersection / len(query_terms)) + exact_bonus + phrase_bonus


def _top_candidates(
    candidates: Sequence[_Candidate],
    limit: int,
) -> list[_Candidate]:
    return sorted(
        candidates,
        key=lambda item: (
            -item.score,
            item.chunk.source_path,
            item.chunk.start_line,
            item.chunk.chunk_id,
        ),
    )[:limit]


def _candidate_to_hit(candidate: _Candidate) -> RAGHit:
    return RAGHit(
        chunk_id=candidate.chunk.chunk_id,
        source_path=candidate.chunk.source_path,
        start_line=candidate.chunk.start_line,
        end_line=candidate.chunk.end_line,
        score=round(candidate.score, 6),
        lexical_score=round(candidate.lexical_score, 6),
        semantic_score=round(candidate.semantic_score, 6),
        reranker_score=round(candidate.reranker_score, 6),
        text=candidate.chunk.text,
        metadata=dict(candidate.chunk.metadata),
        relation_score=round(candidate.relation_score, 6),
    )


def _merge_hits(
    first: Sequence[RAGHit],
    second: Sequence[RAGHit],
) -> list[RAGHit]:
    merged: dict[str, RAGHit] = {hit.chunk_id: hit for hit in first}
    for hit in second:
        previous = merged.get(hit.chunk_id)
        if previous is None or hit.score > previous.score:
            merged[hit.chunk_id] = hit
    return sorted(
        merged.values(),
        key=lambda hit: (
            -hit.score,
            hit.source_path,
            hit.start_line,
            hit.chunk_id,
        ),
    )


def _finalize_hits(
    hits: Sequence[RAGHit],
    *,
    route: str,
    limit: int,
) -> list[RAGHit]:
    ordered = _merge_hits((), hits)
    if route != "global_project":
        return ordered[:limit]
    # Global questions benefit from source diversity instead of many adjacent
    # chunks from the first large file. This is deterministic round-robin.
    by_source: dict[str, list[RAGHit]] = {}
    source_order: list[str] = []
    for hit in ordered:
        if hit.source_path not in by_source:
            by_source[hit.source_path] = []
            source_order.append(hit.source_path)
        by_source[hit.source_path].append(hit)
    diversified: list[RAGHit] = []
    offset = 0
    while len(diversified) < limit:
        added = False
        for source in source_order:
            source_hits = by_source[source]
            if offset < len(source_hits):
                diversified.append(source_hits[offset])
                added = True
                if len(diversified) >= limit:
                    break
        if not added:
            break
        offset += 1
    return diversified


def _result_with_receipt(
    query: str,
    *,
    route: str,
    hits: Sequence[RAGHit],
    corrected_query: str | None,
    correction_applied: bool,
    lexical_backend: str,
    semantic: bool,
    rerank: bool,
    candidates_considered: int,
    relation_expansions: int,
    covered: tuple[str, ...],
    missing: tuple[str, ...],
    coverage: float,
    required_metadata: dict[str, Any],
    warnings: Sequence[str],
) -> RAGSearchResult:
    top_score = max((hit.score for hit in hits), default=0.0)
    normalized_top = top_score / (1.0 + max(0.0, top_score))
    relevance = round((0.6 * coverage) + (0.4 * normalized_top), 6)
    receipt = RAGSearchReceipt(
        schema_version="mmm/rag-search-receipt-v1",
        query=query,
        route=route,
        corrected_query=corrected_query,
        correction_applied=correction_applied,
        lexical_backend=lexical_backend,
        semantic_requested=semantic,
        semantic_used=semantic,
        rerank_requested=rerank,
        rerank_used=rerank and bool(hits),
        candidates_considered=candidates_considered,
        relation_expansions=relation_expansions,
        result_count=len(hits),
        query_terms=_meaningful_terms(query),
        covered_terms=covered,
        missing_terms=missing,
        coverage_score=coverage,
        relevance_score=relevance,
        required_metadata=dict(required_metadata),
        warnings=tuple(warnings),
    )
    return RAGSearchResult(hits=tuple(hits), receipt=receipt)


def _empty_result(
    query: str,
    *,
    route: str,
    semantic: bool,
    rerank: bool,
    required_metadata: dict[str, Any],
    lexical_backend: str,
    warning: str,
) -> RAGSearchResult:
    terms = _meaningful_terms(query)
    return RAGSearchResult(
        hits=(),
        receipt=RAGSearchReceipt(
            schema_version="mmm/rag-search-receipt-v1",
            query=query,
            route=route,
            corrected_query=None,
            correction_applied=False,
            lexical_backend=lexical_backend,
            semantic_requested=semantic,
            semantic_used=False,
            rerank_requested=rerank,
            rerank_used=False,
            candidates_considered=0,
            relation_expansions=0,
            result_count=0,
            query_terms=terms,
            covered_terms=(),
            missing_terms=terms,
            coverage_score=0.0,
            relevance_score=0.0,
            required_metadata=dict(required_metadata),
            warnings=(warning,),
        ),
    )


def _chunk_from_row(
    row: sqlite3.Row | Sequence[Any],
    metadata: dict[str, Any],
) -> RAGChunk:
    if isinstance(row, sqlite3.Row):
        embedding_raw = row["embedding"]
        return RAGChunk(
            chunk_id=str(row["chunk_id"]),
            source_path=str(row["source_path"]),
            text=str(row["text"]),
            start_line=int(row["start_line"]),
            end_line=int(row["end_line"]),
            sha256=str(row["sha256"]),
            metadata=dict(metadata),
            embedding=tuple(
                float(value) for value in json.loads(embedding_raw or "[]")
            ),
        )
    raise TypeError("SQLite RAG rows must use sqlite3.Row.")


def _path_aliases(path: str) -> tuple[str, ...]:
    normalized = _normalize_relation_path(path)
    parts = [part for part in normalized.split("/") if part]
    aliases = {normalized}
    for offset in range(len(parts)):
        aliases.add("/".join(parts[offset:]))
    return tuple(sorted(aliases))


def _normalize_relation_path(path: str) -> str:
    return path.replace("\\", "/").strip("/").casefold()


def _escape_like(value: str) -> str:
    return value.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _meaningful_terms(text: str) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for term in _tokens(text):
        if term in _STOPWORDS or term in seen:
            continue
        if len(term) < 2 and not term.isdigit():
            continue
        seen.add(term)
        result.append(term)
    return tuple(result)


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN.finditer(text)]


def _cosine(
    left: Sequence[float] | None,
    right: Sequence[float],
) -> float:
    if left is None or len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _is_sqlite(path: Path) -> bool:
    with path.open("rb") as source:
        return source.read(len(_SQLITE_HEADER)) == _SQLITE_HEADER


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()
