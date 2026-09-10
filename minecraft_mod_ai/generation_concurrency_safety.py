from __future__ import annotations

"""Thread-safety boundaries for stateful generation helpers shared by the DAG runner."""

import threading
from collections.abc import Callable
from functools import wraps
from typing import Any

_INSTALLED = False
_INIT_LOCK = threading.RLock()
_CUSTOM_LOCK_ATTR = "_mmm_custom_generation_lock"
_INDEX_LOCK_ATTR = "_mmm_project_index_lock"


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
        "update_files",
        "write_manifest",
        "manifest",
        "manifest_receipt",
        "select",
        "select_page",
    ):
        wrap(method_name)


def _install_exact_anchor_fallback(work_graph_module: Any) -> None:
    """Give unscoped writers a conservative collision key.

    Explicit exclusive owned_anchors are the precise file/registry collision domain.
    A module without such provenance must not silently become concurrent, so it gets
    a stage-scoped fallback anchor. This preserves fail-closed behavior while letting
    correctly scoped modules in the same stage run independently.
    """
    current = work_graph_module._exclusive_anchor_keys
    if getattr(current, "_mmm_unscoped_fallback", False):
        return

    @wraps(current)
    def exclusive_anchor_keys(module: Any) -> tuple[str, ...]:
        exact = tuple(current(module))
        if exact:
            return exact
        stage = work_graph_module._module_stage(module)
        if stage in {"content", "system", "entity"}:
            return (f"mmm://unscoped-stage/{stage}",)
        return ()

    exclusive_anchor_keys._mmm_unscoped_fallback = True  # type: ignore[attr-defined]
    exclusive_anchor_keys.__wrapped__ = current  # type: ignore[attr-defined]
    work_graph_module._exclusive_anchor_keys = exclusive_anchor_keys


def _replace_stage_locks_with_anchor_fencing(work_graph_module: Any) -> None:
    """Replace global stage critical sections with exact WorkGraph collision edges."""
    from . import scheduler_parallel_safety_contract as scheduler_safety

    anchor_resolver = getattr(work_graph_module, "_exclusive_anchor_keys", None)
    if not callable(anchor_resolver) or not getattr(anchor_resolver, "_mmm_unscoped_fallback", False):
        return

    # claim_ready reads these globals dynamically. Emptying them removes the old
    # stage-wide admission barrier; WorkGraph dependency edges now serialize only
    # nodes that own the same exact anchor (or the conservative unscoped fallback).
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
