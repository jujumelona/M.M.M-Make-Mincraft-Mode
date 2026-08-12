from __future__ import annotations

import hashlib
import threading
from functools import wraps
from pathlib import Path
from typing import Any

_CACHE_LOCK = threading.RLock()
_CACHE_LIMIT = 32
_CACHE: dict[str, tuple[str, str]] = {}
_SPECIAL_NAMES = frozenset(
    {"build.gradle", "settings.gradle", "gradle.properties", "fabric.mod.json"}
)


def _metadata_signature(project_index_module: Any, root: Path) -> str:
    """Fingerprint source metadata without reading or tokenizing file contents."""

    digest = hashlib.sha256()
    root = root.resolve()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        if any(part in project_index_module._IGNORED_PARTS for part in relative.parts):
            continue
        suffix = path.suffix.lower()
        if suffix not in project_index_module._TEXT_SUFFIXES and path.name not in _SPECIAL_NAMES:
            continue
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
    """Avoid repeated full source reads when the project tree is unchanged.

    CompleteProductionOrchestrator asks for the same manifest commitment several
    times between generation, deterministic validation, JDT and build/package gates.
    Constructing a fresh ProjectIndex on every call rereads and tokenizes the entire
    source tree. A metadata-only signature is enough to prove whether the expensive
    content commitment must be recomputed. If anything changed, the original full
    content hash remains authoritative.
    """

    cls = orchestrator_module.CompleteProductionOrchestrator
    current = cls._project_manifest_hash
    if getattr(current, "_mmm_metadata_manifest_cache", False):
        return

    @wraps(current)
    def cached_manifest_hash(self: Any, project_root: Path) -> str:
        root = Path(project_root).expanduser().resolve()
        root_key = str(root)
        before = _metadata_signature(project_index_module, root)
        with _CACHE_LOCK:
            cached = _CACHE.get(root_key)
        if cached is not None and cached[0] == before:
            return cached[1]

        # Full ProjectIndex scan/content hashing is still the source of truth whenever
        # metadata differs. Recheck metadata after it finishes; if writers raced the
        # scan, do not cache that result for a later call.
        value = str(current(self, root))
        after = _metadata_signature(project_index_module, root)
        if before == after:
            _bounded_store(root_key, after, value)
        return value

    cached_manifest_hash._mmm_metadata_manifest_cache = True  # type: ignore[attr-defined]
    cached_manifest_hash.__wrapped__ = current  # type: ignore[attr-defined]
    cls._project_manifest_hash = cached_manifest_hash


__all__ = ["install"]
