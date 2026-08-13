from __future__ import annotations

import hashlib
import json
import threading
from functools import wraps
from pathlib import Path
from typing import Any

_INDEX_BUILD_LOCKS = tuple(threading.RLock() for _ in range(64))
_BUILT_TARGETS_LOCK = threading.RLock()
_BUILT_TARGET_FINGERPRINTS: dict[Path, str] = {}
_MISSING = object()


def _index_lock(path: Path) -> threading.RLock:
    digest = hashlib.sha256(str(path).encode("utf-8")).digest()
    slot = int.from_bytes(digest[:2], "big") % len(_INDEX_BUILD_LOCKS)
    return _INDEX_BUILD_LOCKS[slot]


def _build_fingerprint(
    roots: Any,
    *,
    metadata: dict[str, Any],
    semantic: bool,
) -> str:
    """Identify one logical index build without inspecting mutable output state."""
    if isinstance(roots, (list, tuple)):
        root_values = [str(value) for value in roots]
    else:
        # ProductionToolService accepts a Sequence. Keep compatibility with narrow
        # test doubles without consuming an arbitrary one-shot iterable.
        root_values = [str(roots)]
    payload = {
        "roots": root_values,
        "metadata": metadata,
        "semantic": bool(semantic),
    }
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(rendered.encode("utf-8")).hexdigest()


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
            # Serialize only conflicting canonical targets. If another caller asks
            # for the exact same logical build, the fresh artifact is rechecked
            # instead of rebuilt. A changed source revision/metadata/roots is a real
            # refresh and is therefore allowed to replace the index in place.
            target = self._resolve(index_path)
            fingerprint = _build_fingerprint(
                roots,
                metadata=metadata,
                semantic=semantic,
            )
            with _index_lock(target):
                with _BUILT_TARGETS_LOCK:
                    previous = _BUILT_TARGET_FINGERPRINTS.get(target)
                if previous == fingerprint and target.exists():
                    raise FileExistsError(
                        "Identical RAG index build already completed and must be "
                        f"rechecked: {target}"
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
                        _BUILT_TARGET_FINGERPRINTS[target] = fingerprint
                    else:
                        _BUILT_TARGET_FINGERPRINTS.pop(target, None)
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
