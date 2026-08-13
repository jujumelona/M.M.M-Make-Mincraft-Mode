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
        existed_before_wait = target.exists()
        with _index_lock(target):
            # If this caller observed no index before waiting but the same canonical
            # target appeared while it was queued, another builder won the race. Do
            # not immediately rebuild that fresh artifact: report the collision so
            # the caller rechecks/consumes it. Pre-existing live indexes remain
            # replaceable derived artifacts and may be intentionally refreshed.
            if not existed_before_wait and target.exists():
                raise FileExistsError(
                    f"RAG index was created by a concurrent builder: {target}"
                )
            return current(
                self,
                roots,
                index_path=index_path,
                metadata=metadata,
                semantic=semantic,
            )

    index_project_rag._mmm_path_serialized_rag_build = True
    service_cls.index_project_rag = index_project_rag
