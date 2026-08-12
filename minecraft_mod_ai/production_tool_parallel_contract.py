from __future__ import annotations

import hashlib
import threading
from functools import wraps
from pathlib import Path
from typing import Any

_INDEX_BUILD_LOCKS = tuple(threading.RLock() for _ in range(64))


def _index_lock(path: Path) -> threading.RLock:
    digest = hashlib.sha256(str(path).encode("utf-8")).digest()
    slot = int.from_bytes(digest[:2], "big") % len(_INDEX_BUILD_LOCKS)
    return _INDEX_BUILD_LOCKS[slot]


def install(production_tools_module: Any) -> None:
    service_cls = production_tools_module.ProductionToolService
    current = service_cls.index_project_rag
    if getattr(current, "_mmm_path_serialized_rag_build", False):
        return

    @wraps(current)
    def index_project_rag(
        self: Any,
        roots: Any,
        *,
        index_path: str = "rag/project-index.json",
        metadata: dict[str, Any],
        semantic: bool = False,
    ) -> dict[str, Any]:
        # _resolve is side-effect free. Resolve before taking the stripe so only
        # equivalent canonical output paths serialize with each other.
        target = self._resolve(index_path)
        with _index_lock(target):
            # The original method also checks via _new_file(), but that check was
            # previously outside any mutual exclusion. Repeat it inside the lock to
            # close the check/build/atomic-replace TOCTOU window.
            if target.exists():
                raise FileExistsError(target)
            return current(
                self,
                roots,
                index_path=index_path,
                metadata=metadata,
                semantic=semantic,
            )

    index_project_rag._mmm_path_serialized_rag_build = True
    service_cls.index_project_rag = index_project_rag
