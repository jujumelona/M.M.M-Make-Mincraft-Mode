from __future__ import annotations

"""Thread-safety boundaries for stateful generation helpers shared by the DAG runner."""

import threading
from collections.abc import Callable
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

_INSTALLED = False
_INIT_LOCK = threading.RLock()
_CUSTOM_LOCK_ATTR = "_mmm_custom_generation_lock"
_INDEX_LOCK_ATTR = "_mmm_project_index_lock"
_PATH_KEYS = frozenset({
    "path", "target_path", "source_path", "output_path", "file", "target_file",
    "source_file", "manifest_path", "registry_path", "resource_path",
})
_PATH_LIST_KEYS = frozenset({
    "paths", "target_paths", "source_paths", "output_paths", "files",
    "touched_paths", "written_files",
})


def _lock_for(instance: Any, attribute: str) -> threading.RLock:
    lock = getattr(instance, attribute, None)
    if lock is not None:
        return lock
    with _INIT_LOCK:
        lock = getattr(instance, attribute, None)
        if lock is None:
            lock = threading.RLock()
            setattr(instance, attribute, lock)
        return lock


def _install_custom_generator_lock(custom_module_generator_module: Any) -> None:
    cls = custom_module_generator_module.CustomModuleGenerator
    current = cls.generate
    if getattr(current, "_mmm_instance_generation_serialized", False):
        return

    @wraps(current)
    def generate(self: Any, *args: Any, **kwargs: Any):
        with _lock_for(self, _CUSTOM_LOCK_ATTR):
            return current(self, *args, **kwargs)

    generate._mmm_instance_generation_serialized = True  # type: ignore[attr-defined]
    generate.__wrapped__ = current  # type: ignore[attr-defined]
    cls.generate = generate


def _install_project_index_snapshot_lock(project_index_module: Any) -> None:
    cls = project_index_module.ProjectIndex

    def wrap(name: str) -> None:
        current: Callable[..., Any] = getattr(cls, name)
        if getattr(current, "_mmm_snapshot_locked", False):
            return

        @wraps(current)
        def locked(self: Any, *args: Any, **kwargs: Any):
            with _lock_for(self, _INDEX_LOCK_ATTR):
                return current(self, *args, **kwargs)

        locked._mmm_snapshot_locked = True  # type: ignore[attr-defined]
        locked.__wrapped__ = current  # type: ignore[attr-defined]
        setattr(cls, name, locked)

    for method_name in (
        "update_files", "write_manifest", "manifest", "manifest_receipt",
        "select", "select_page",
    ):
        wrap(method_name)


def _safe_path_anchor(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    rendered = value.strip().replace("\\", "/")
    if not rendered or rendered.startswith("/"):
        return None
    path = PurePosixPath(rendered)
    if any(part in {"", ".", ".."} for part in path.parts):
        return None
    return "file://" + path.as_posix()


def _inferred_config_anchors(config: Any) -> tuple[str, ...]:
    """Extract only high-confidence filesystem collision keys from reviewed config."""
    found: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized = str(key).strip().casefold()
                if normalized in _PATH_KEYS:
                    anchor = _safe_path_anchor(nested)
                    if anchor:
                        found.add(anchor)
                elif normalized in _PATH_LIST_KEYS and isinstance(nested, (list, tuple)):
                    for item in nested:
                        anchor = _safe_path_anchor(item)
                        if anchor:
                            found.add(anchor)
                if isinstance(nested, (dict, list, tuple)):
                    visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)

    visit(config)
    return tuple(sorted(found))


def _builtin_shared_anchors(module: Any, stage: str) -> tuple[str, ...]:
    """Model shared files that built-in generators are known to read/merge/rewrite."""
    kind = str(getattr(module, "kind", ""))
    if stage == "content" and kind != "integration":
        # ExtendedContentGenerator merges its catalog/language files and rewrites the
        # shared GeneratedExtendedContent registrar plus the main initializer binding.
        return ("mmm://builtin/content/shared-registration",)
    if stage == "system":
        # Every system pack emits the common persistent/config classes and edits the
        # shared main initializer. Different pack-specific files alone are not enough.
        return ("mmm://builtin/system/shared-runtime",)
    if stage == "entity":
        # GeckoLib generation rewrites shared entity registrars, dependency metadata,
        # fabric.mod.json and main/client entrypoint bindings.
        return ("mmm://builtin/entity/shared-runtime",)
    return ()


def _install_exact_anchor_fallback(work_graph_module: Any) -> None:
    """Resolve explicit, inferred and built-in collision domains; fail closed last."""
    current = work_graph_module._exclusive_anchor_keys
    if getattr(current, "_mmm_unscoped_fallback", False):
        return

    @wraps(current)
    def exclusive_anchor_keys(module: Any) -> tuple[str, ...]:
        explicit = tuple(current(module))
        stage = work_graph_module._module_stage(module)
        config = getattr(module, "config", {})
        inferred = _inferred_config_anchors(config)
        shared = _builtin_shared_anchors(module, stage)
        anchors = tuple(dict.fromkeys((*explicit, *inferred, *shared)))
        if anchors:
            return anchors
        if stage in {"content", "system", "entity"}:
            return (f"mmm://unscoped-stage/{stage}",)
        return ()

    exclusive_anchor_keys._mmm_unscoped_fallback = True  # type: ignore[attr-defined]
    exclusive_anchor_keys._mmm_config_anchor_inference = True  # type: ignore[attr-defined]
    exclusive_anchor_keys.__wrapped__ = current  # type: ignore[attr-defined]
    work_graph_module._exclusive_anchor_keys = exclusive_anchor_keys


def _replace_stage_locks_with_anchor_fencing(work_graph_module: Any) -> None:
    """Replace global stage critical sections with WorkGraph collision edges."""
    from . import scheduler_parallel_safety_contract as scheduler_safety

    anchor_resolver = getattr(work_graph_module, "_exclusive_anchor_keys", None)
    if not callable(anchor_resolver) or not getattr(anchor_resolver, "_mmm_unscoped_fallback", False):
        return
    scheduler_safety._STAGE_WRITE_LOCKS.clear()
    scheduler_safety._SERIAL_CPU_STAGES = ()


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    with _INIT_LOCK:
        if _INSTALLED:
            return
        from . import custom_module_generator, project_index, work_graph

        _install_custom_generator_lock(custom_module_generator)
        _install_project_index_snapshot_lock(project_index)
        _install_exact_anchor_fallback(work_graph)
        _replace_stage_locks_with_anchor_fencing(work_graph)
        _INSTALLED = True


__all__ = ["install"]
