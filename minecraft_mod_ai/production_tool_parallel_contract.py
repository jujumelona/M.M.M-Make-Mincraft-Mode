from __future__ import annotations

import threading
from functools import wraps
from pathlib import Path
from typing import Any

_TARGET_LOCKS_GUARD = threading.RLock()
_TARGET_LOCKS: dict[Path, threading.RLock] = {}
_MISSING = object()


def _target_lock(path: Path) -> threading.RLock:
    """Return one stable lock per canonical RAG output target."""
    canonical = path.expanduser().resolve()
    with _TARGET_LOCKS_GUARD:
        lock = _TARGET_LOCKS.get(canonical)
        if lock is None:
            lock = threading.RLock()
            _TARGET_LOCKS[canonical] = lock
        return lock


def _index_lock(path: Path) -> threading.RLock:
    """Compatibility name for callers/tests using the original striped RAG API."""
    return _target_lock(path)


def install(production_tools_module: Any) -> None:
    service_cls = production_tools_module.ProductionToolService
    current = service_cls.index_project_rag
    if not getattr(current, "_mmm_path_serialized_rag_build", False):

        @wraps(current)
        def index_project_rag(
            self: Any,
            roots: Any,
            *,
            index_path: str = "rag/project-index.json",
            metadata: dict[str, Any],
            semantic: bool = False,
        ) -> dict[str, Any]:
            # Only overlapping builds of the same canonical target conflict. A later,
            # sequential call is an intentional refresh and is allowed even when its
            # roots/metadata are identical to the previous build. Different targets
            # never share a lock and therefore remain fully parallel.
            target = self._resolve(index_path)
            lock = _target_lock(target)
            waited_for_builder = not lock.acquire(blocking=False)
            if waited_for_builder:
                lock.acquire()
            try:
                if waited_for_builder and target.exists():
                    raise FileExistsError(
                        "RAG index was completed by an overlapping builder and must be "
                        f"rechecked before rebuilding: {target}"
                    )
                return current(
                    self,
                    roots,
                    index_path=index_path,
                    metadata=metadata,
                    semantic=semantic,
                )
            finally:
                lock.release()

        index_project_rag._mmm_path_serialized_rag_build = True
        service_cls.index_project_rag = index_project_rag

    _install_custom_generator_router_compat()


def _install_custom_generator_router_compat() -> None:
    return
