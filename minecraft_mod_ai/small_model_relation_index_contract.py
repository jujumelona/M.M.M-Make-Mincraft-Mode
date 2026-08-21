from __future__ import annotations

import hashlib
import json
import sqlite3
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any, Sequence

from .small_model_rag_relations import derive_relations

_REUSE_LOCK = RLock()
_REUSE_RESULTS: dict[tuple[Any, ...], dict[str, Any]] = {}
_REUSE_LIMIT = 64
_SQLITE_HEADER = b"SQLite format 3\x00"
_RELATION_METADATA_KEYS = frozenset({"relations", "relation_count"})


def _metadata_key(metadata: dict[str, Any]) -> str:
    filtered = {
        key: value
        for key, value in metadata.items()
        if key not in _RELATION_METADATA_KEYS
    }
    return json.dumps(
        filtered,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _reuse_key(
    target: Path,
    metadata: dict[str, Any],
    semantic: bool,
) -> tuple[Any, ...] | None:
    try:
        stat = target.stat()
    except OSError:
        return None
    if not target.is_file() or target.is_symlink():
        return None
    return (
        str(target),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        _metadata_key(metadata),
        bool(semantic),
    )


def _cache_get(key: tuple[Any, ...] | None) -> dict[str, Any] | None:
    if key is None:
        return None
    with _REUSE_LOCK:
        cached = _REUSE_RESULTS.get(key)
        if cached is None:
            return None
        result = dict(cached)
    result["reused"] = True
    result["reuse_reason"] = "exact_project_snapshot"
    return result


def _cache_put(
    target: Path,
    metadata: dict[str, Any],
    semantic: bool,
    result: dict[str, Any],
) -> None:
    key = _reuse_key(target, metadata, semantic)
    if key is None:
        return
    with _REUSE_LOCK:
        if len(_REUSE_RESULTS) >= _REUSE_LIMIT and key not in _REUSE_RESULTS:
            _REUSE_RESULTS.pop(next(iter(_REUSE_RESULTS)))
        _REUSE_RESULTS[key] = dict(result)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _existing_snapshot_result(
    target: Path,
    metadata: dict[str, Any],
    semantic: bool,
) -> dict[str, Any] | None:
    key = _reuse_key(target, metadata, semantic)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if key is None:
        return None
    try:
        with target.open("rb") as stream:
            if stream.read(len(_SQLITE_HEADER)) != _SQLITE_HEADER:
                return None
        with sqlite3.connect(str(target)) as connection:
            meta = {
                str(row[0]): str(row[1])
                for row in connection.execute(
                    "SELECT key, value FROM index_meta ORDER BY key"
                )
            }
    except (OSError, sqlite3.DatabaseError):
        return None
    try:
        stored_metadata = json.loads(meta.get("metadata", "{}"))
    except (TypeError, ValueError):
        return None
    if not isinstance(stored_metadata, dict):
        return None
    for field, value in metadata.items():
        if field in _RELATION_METADATA_KEYS:
            continue
        if stored_metadata.get(field) != value:
            return None
    if (meta.get("semantic_embeddings") == "1") != bool(semantic):
        return None
    if not str(metadata.get("source_commit", "")).strip():
        return None

    result = {
        "schema_version": "mmm/rag-build-result-v1",
        "index_schema_version": meta.get(
            "schema_version", "mmm/project-rag-index-v2"
        ),
        "index_backend": "sqlite",
        "lexical_backend": (
            "sqlite_fts5" if meta.get("fts5") == "1" else "deterministic_scan"
        ),
        "index_path": str(target),
        "files_indexed": int(meta.get("files_indexed", "0") or 0),
        "chunks_indexed": int(meta.get("chunks_indexed", "0") or 0),
        "semantic_embeddings": bool(semantic),
        "embedding_dimensions": int(meta.get("embedding_dimensions", "0") or 0),
        "index_sha256": _sha256(target),
        "reused": True,
        "reuse_reason": "exact_project_snapshot",
    }
    _cache_put(target, metadata, semantic, result)
    return result


def install(production_tools_module: Any) -> None:
    cls = production_tools_module.ProductionToolService
    current = cls.index_project_rag
    if getattr(current, "_mmm_dependency_relations", False):
        return
    # small_model_max_agent wraps the parallel-safe canonical indexer and forces
    # semantic=True for project-local repair indexes. Its __wrapped__ target is the
    # already-reviewed parallel-safe lexical implementation, so this contract can
    # deliberately bypass only that one expensive semantic override while retaining
    # relation derivation, exact-snapshot reuse, and the underlying build lock.
    fallback = getattr(current, "__wrapped__", None)

    @wraps(current)
    def indexed(
        self: Any,
        roots: Sequence[str],
        *,
        index_path: str = "rag/project-index.json",
        metadata: dict[str, Any],
        semantic: bool = False,
    ):
        target = self._resolve(index_path)
        reused = _existing_snapshot_result(target, metadata, semantic)
        if reused is not None:
            return reused

        enriched = dict(metadata)
        relations = derive_relations([self._existing_path(root) for root in roots])
        if relations:
            enriched["relations"] = relations
            enriched["relation_count"] = len(relations)
        repair_like = bool(enriched.get("source_commit")) and str(
            enriched.get("license", "")
        ) == "project-local"

        # Caller intent wins. A normal repair index is lexical+graph and must not
        # silently instantiate the 0.6B CPU embedding model. Explicit semantic=True
        # still goes through the full semantic owner unchanged.
        if repair_like and not semantic and callable(fallback):
            result = dict(
                fallback(
                    self,
                    roots,
                    index_path=index_path,
                    metadata=enriched,
                    semantic=False,
                )
            )
            _cache_put(target, metadata, False, result)
            return result

        try:
            result = dict(
                current(
                    self,
                    roots,
                    index_path=index_path,
                    metadata=enriched,
                    semantic=semantic,
                )
            )
        except Exception:
            if not callable(fallback) or not repair_like:
                raise
            result = dict(
                fallback(
                    self,
                    roots,
                    index_path=index_path,
                    metadata=enriched,
                    semantic=False,
                )
            )
            semantic = False
        _cache_put(target, metadata, semantic, result)
        return result

    indexed._mmm_dependency_relations = True  # type: ignore[attr-defined]
    indexed._mmm_exact_snapshot_reuse = True  # type: ignore[attr-defined]
    # @wraps copies marker attributes from the wrapped semantic-forcing function.
    # Remove that inherited marker: this wrapper now owns the repair semantic policy
    # and downstream efficiency layers must not try to unwrap it again.
    indexed.__dict__.pop("_mmm_small_model_semantic_repair_index", None)
    indexed.__wrapped__ = current  # type: ignore[attr-defined]
    cls.index_project_rag = indexed


__all__ = ["install"]
