from __future__ import annotations

import hashlib
import os
import threading
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

_CACHE_LOCK = threading.RLock()
_CACHE_LIMIT = 32
_CACHE: dict[str, tuple[str, str]] = {}
_EXECUTION_CACHE: ContextVar[dict[str, str] | None] = ContextVar(
    "mmm_execution_project_manifest_cache",
    default=None,
)
_SPECIAL_NAMES = frozenset(
    {"build.gradle", "settings.gradle", "gradle.properties", "fabric.mod.json"}
)


def _metadata_signature(project_index_module: Any, root: Path) -> str:
    """Fingerprint source metadata without walking ignored build/cache subtrees."""

    digest = hashlib.sha256()
    root = root.resolve()
    ignored_parts = project_index_module._IGNORED_PARTS
    text_suffixes = project_index_module._TEXT_SUFFIXES
    candidates: list[Path] = []

    for current_root, dir_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        # rglob() discovers the full tree before the ignored-part filter runs. Large
        # build/cache directories therefore made even a cache-hit check scale with
        # generated output. Prune those directories before descent instead.
        dir_names[:] = [name for name in dir_names if name not in ignored_parts]
        current = Path(current_root)
        for name in file_names:
            path = current / name
            if path.is_symlink():
                continue
            if path.suffix.lower() not in text_suffixes and name not in _SPECIAL_NAMES:
                continue
            candidates.append(path)

    # Preserve the previous path ordering so unchanged source metadata produces the
    # exact same signature after this host-side traversal optimization.
    for path in sorted(candidates, key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        stat = path.stat()
        record = (
            relative.as_posix(),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
        rendered = repr(record).encode("utf-8")
        digest.update(len(rendered).to_bytes(8, "big"))
        digest.update(rendered)
    return digest.hexdigest()


def _bounded_store(root: str, signature: str, value: str) -> None:
    with _CACHE_LOCK:
        _CACHE[root] = (signature, value)
        while len(_CACHE) > _CACHE_LIMIT:
            _CACHE.pop(next(iter(_CACHE)))


def install(orchestrator_module: Any, project_index_module: Any) -> None:
    """Avoid repeated full source reads and scans for one authoritative snapshot.

    The global cache remains metadata-validated across independent calls. During one
    ``execute()`` invocation, the first post-generation source commitment is also the
    exact input to deterministic validation, JDT and build/package checkpoints. Those
    calls occur before any repair action can mutate source, so the already-authoritative
    value can be reused without rescanning the tree. The execution cache is ContextVar-
    scoped so concurrent runs never share an unvalidated value.
    """

    cls = orchestrator_module.CompleteProductionOrchestrator
    current = cls._project_manifest_hash
    if not getattr(current, "_mmm_metadata_manifest_cache", False):

        @wraps(current)
        def cached_manifest_hash(self: Any, project_root: Path) -> str:
            root = Path(project_root).expanduser().resolve()
            root_key = str(root)

            execution_cache = _EXECUTION_CACHE.get()
            if execution_cache is not None:
                execution_value = execution_cache.get(root_key)
                if execution_value is not None:
                    return execution_value

            before = _metadata_signature(project_index_module, root)
            with _CACHE_LOCK:
                cached = _CACHE.get(root_key)
            if cached is not None and cached[0] == before:
                value = cached[1]
                if execution_cache is not None:
                    execution_cache[root_key] = value
                return value

            # Full ProjectIndex scan/content hashing is still the source of truth whenever
            # metadata differs. Recheck metadata after it finishes; if writers raced the
            # scan, do not cache that result for a later independent call.
            value = str(current(self, root))
            after = _metadata_signature(project_index_module, root)
            if before == after:
                _bounded_store(root_key, after, value)
                if execution_cache is not None:
                    execution_cache[root_key] = value
            return value

        cached_manifest_hash._mmm_metadata_manifest_cache = True  # type: ignore[attr-defined]
        cached_manifest_hash._mmm_execution_manifest_cache = True  # type: ignore[attr-defined]
        cached_manifest_hash.__wrapped__ = current  # type: ignore[attr-defined]
        cls._project_manifest_hash = cached_manifest_hash

    current_execute = getattr(cls, "execute", None)
    if callable(current_execute) and not getattr(
        current_execute,
        "_mmm_execution_manifest_scope",
        False,
    ):

        @wraps(current_execute)
        def execute_with_manifest_scope(self: Any, *args: Any, **kwargs: Any):
            token = _EXECUTION_CACHE.set({})
            try:
                return current_execute(self, *args, **kwargs)
            finally:
                _EXECUTION_CACHE.reset(token)

        execute_with_manifest_scope._mmm_execution_manifest_scope = True  # type: ignore[attr-defined]
        cls.execute = execute_with_manifest_scope


__all__ = ["install"]
