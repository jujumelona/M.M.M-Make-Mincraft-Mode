from __future__ import annotations

import hashlib
import threading
from functools import wraps
from pathlib import Path
from typing import Any

_INDEX_BUILD_LOCKS = tuple(threading.RLock() for _ in range(64))
_MISSING = object()


def _index_lock(path: Path) -> threading.RLock:
    digest = hashlib.sha256(str(path).encode("utf-8")).digest()
    slot = int.from_bytes(digest[:2], "big") % len(_INDEX_BUILD_LOCKS)
    return _INDEX_BUILD_LOCKS[slot]


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

    _install_custom_generator_router_compat()


def _install_custom_generator_router_compat() -> None:
    """Keep narrow test/embedding routers usable without weakening real ModelRouter RAG."""
    from . import custom_module_generator as custom_module_module

    cls = custom_module_module.CustomModuleGenerator
    current = cls.generate
    if getattr(current, "_mmm_lightweight_router_workspace_compat", False):
        return

    @wraps(current)
    def generate(self: Any, *args: Any, **kwargs: Any):
        router = self.router
        old_bind = getattr(router, "bind_agent_workspace", _MISSING)
        old_profile = getattr(router, "profile", _MISSING)
        added_bind = old_bind is _MISSING
        added_profile = old_profile is _MISSING

        # Production ModelRouter already owns both attributes. This compatibility
        # path exists only for deliberately narrow routers used by isolated generator
        # tests/embedders. It does not bypass live code-RAG for production execution.
        if added_bind:
            setattr(
                router,
                "bind_agent_workspace",
                lambda _workspace, *, require_fresh_evidence=False: None,
            )
        if added_profile:
            setattr(router, "profile", "t4_local")
        try:
            return current(self, *args, **kwargs)
        finally:
            if added_bind:
                try:
                    delattr(router, "bind_agent_workspace")
                except AttributeError:
                    pass
            if added_profile:
                try:
                    delattr(router, "profile")
                except AttributeError:
                    pass

    generate._mmm_lightweight_router_workspace_compat = True
    cls.generate = generate
