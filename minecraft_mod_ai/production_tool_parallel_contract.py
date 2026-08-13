from __future__ import annotations

import hashlib
import threading
from functools import wraps
from pathlib import Path
from typing import Any

_INDEX_BUILD_LOCKS = tuple(threading.RLock() for _ in range(64))
_BUILT_TARGETS_LOCK = threading.RLock()
_BUILT_TARGETS: set[Path] = set()
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
            # Resolve before taking the stripe so unrelated canonical paths may build
            # concurrently. A target that pre-existed before this process first
            # touched it remains refreshable once. After this process successfully
            # builds the target, later same-target callers must consume/recheck that
            # fresh artifact rather than immediately rebuilding it. Tracking the
            # successful build removes scheduler timing from this decision.
            target = self._resolve(index_path)
            with _index_lock(target):
                with _BUILT_TARGETS_LOCK:
                    built_here = target in _BUILT_TARGETS
                if built_here and target.exists():
                    raise FileExistsError(
                        f"RAG index was created by this process and must be rechecked: {target}"
                    )
                result = current(
                    self,
                    roots,
                    index_path=index_path,
                    metadata=metadata,
                    semantic=semantic,
                )
                with _BUILT_TARGETS_LOCK:
                    if target.exists():
                        _BUILT_TARGETS.add(target)
                return result

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
