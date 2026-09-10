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
        # CompleteProductionOrchestrator can dispatch multiple LLM work nodes at the
        # same time while reusing one CustomModuleGenerator. The search contract
        # temporarily rebinds self.router/_cached_index/_cached_root, so one instance
        # must never execute two generation calls concurrently. This lock is per
        # generator instance; independent generators remain parallel.
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

    # update_files rebuilds _by_path from a snapshot, so concurrent writers can lose
    # each other's updates. Reads such as select_page derive ranked files and a
    # manifest fingerprint in multiple steps; guarding the full operation prevents a
    # writer from changing the snapshot halfway through that derivation.
    for method_name in (
        "update_files",
        "write_manifest",
        "manifest",
        "manifest_receipt",
        "select",
        "select_page",
    ):
        wrap(method_name)


def _replace_stage_locks_with_anchor_fencing() -> None:
    """Remove coarse stage mutexes after the WorkGraph has exact collision fences.

    WorkGraph construction records each module's exclusive ``owned_anchors`` and
    adds a dependency edge only when two nodes own the same anchor.  The old
    scheduler additionally serialized every content/system/entity node, turning
    unrelated files into one global critical section.  Once exact anchor fencing is
    available, keeping those stage locks only destroys safe parallelism.

    Unknown/unannotated mutation is still protected by the generator's own mutation
    authority/write-scope contracts. Shared ProjectIndex commits remain serialized by
    the snapshot lock above. CustomModuleGenerator remains protected per instance.
    """
    from . import scheduler_parallel_safety_contract as scheduler_safety
    from . import work_graph

    anchor_resolver = getattr(work_graph, "_exclusive_anchor_keys", None)
    if not callable(anchor_resolver):
        # Fail closed on older/incomplete runtimes: retain the coarse locks rather
        # than allowing potentially colliding writes to run concurrently.
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
        from . import custom_module_generator, project_index

        _install_custom_generator_lock(custom_module_generator)
        _install_project_index_snapshot_lock(project_index)
        _replace_stage_locks_with_anchor_fencing()
        _INSTALLED = True


__all__ = ["install"]
