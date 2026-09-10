from __future__ import annotations

"""Thread-safety boundaries for stateful generation helpers shared by the DAG runner."""

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
_EXTENDED_CONTENT_KINDS = frozenset({
    'item', 'block', 'tool', 'weapon', 'armor', 'food', 'crop', 'machine',
    'effect', 'enchantment', 'command', 'recipe', 'advancement', 'loot',
})
_SYSTEM_PACK_BY_KIND = {
    "quest": "quest-system",
    "class": "class-skill-system",
    "skill": "class-skill-system",
    "economy": "economy-shop",
    "shop": "economy-shop",
    "gui": "gui-networking",
    "networking": "gui-networking",
    "party": "party-guild",
    "guild": "party-guild",
}


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


def _project_index_scan_workers(count: int) -> int:
    if count <= 1:
        return 1
    raw = os.environ.get("MMM_PROJECT_INDEX_WORKERS", "").strip()
    if raw:
        try:
            requested = int(raw)
        except ValueError as exc:
            raise ValueError("MMM_PROJECT_INDEX_WORKERS must be a positive integer") from exc
        if requested < 1:
            raise ValueError("MMM_PROJECT_INDEX_WORKERS must be a positive integer")
    else:
        requested = min(8, max(2, os.cpu_count() or 2))
    return min(count, requested)


def _install_project_index_parallel_scan(project_index_module: Any) -> None:
    """Parallelize independent initial file hashing/tokenization without reordering."""
    cls = project_index_module.ProjectIndex
    current = cls._scan
    if getattr(current, "_mmm_bounded_parallel_scan", False):
        return

    @wraps(current)
    def scan(self: Any):
        paths = tuple(
            path
            for path in sorted(self.root.rglob("*"))
            if path.is_file() and not path.is_symlink()
        )
        configured = bool(os.environ.get("MMM_PROJECT_INDEX_WORKERS", "").strip())
        # Thread startup dominates tiny source trees. An explicit host setting opts
        # into parallel execution even for small projects, which also makes the
        # concurrency boundary directly testable.
        if len(paths) < 16 and not configured:
            return current(self)
        workers = _project_index_scan_workers(len(paths))
        if workers <= 1:
            return current(self)

        def index_one(path: Any):
            relative = path.relative_to(self.root)
            return self._indexed_file(relative.as_posix(), path)

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="mmm-project-index",
        ) as pool:
            # Executor.map preserves the deterministic lexical path order while file
            # reads, hashes and tokenization happen concurrently.
            indexed = tuple(pool.map(index_one, paths))
        return tuple(item for item in indexed if item is not None)

    scan._mmm_bounded_parallel_scan = True  # type: ignore[attr-defined]
    scan.__wrapped__ = current  # type: ignore[attr-defined]
    cls._scan = scan


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
    """Model only generators whose pre-commit state merge is genuinely shared.

    GeckoLib entity generation deliberately has no stage-wide anchor here: geometry,
    animations, Java sources and per-entity directory records are entity-local. Its
    remaining shared writes are already serialized by project_edit atomic helpers.

    System state merge is pack-local, not stage-global. Modules in the same pack must
    remain ordered because they read/merge that pack's record directory, while distinct
    packs can prepare in parallel and meet only at the short shared-file commit lock.
    """
    kind = str(getattr(module, "kind", ""))
    if stage == "content" and kind in _EXTENDED_CONTENT_KINDS:
        module_id = str(getattr(module, "module_id", "")).strip()
        if module_id:
            return (f"mmm://builtin/content/module/{module_id}",)
    if stage == "system":
        pack = _SYSTEM_PACK_BY_KIND.get(kind)
        if pack:
            return (f"mmm://builtin/system/pack/{pack}",)
        return ("mmm://builtin/system/unclassified",)
    return ()


def _configure_pipeline_granularity() -> None:
    """Choose generation units that match each generator's actual state boundary.

    Content and entity work remain singleton by default so independent artifacts and
    reviews stay exposed to the DAG. Built-in system validation and record generation
    are pack-aggregate operations, so system nodes use the Java shard budget instead
    of forcing one cumulative directory scan per individual module.
    """
    os.environ.setdefault("MMM_CONTENT_PIPELINE_SHARD_SIZE", "1")
    os.environ.setdefault(
        "MMM_SYSTEM_PIPELINE_SHARD_SIZE",
        os.environ.get("MMM_JAVA_SHARD_SIZE", "48").strip() or "48",
    )
    os.environ.setdefault("MMM_ENTITY_PIPELINE_SHARD_SIZE", "1")


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
        if stage == "content":
            return ("mmm://unscoped-stage/content",)
        return ()

    exclusive_anchor_keys._mmm_unscoped_fallback = True  # type: ignore[attr-defined]
    exclusive_anchor_keys._mmm_config_anchor_inference = True  # type: ignore[attr-defined]
    exclusive_anchor_keys.__wrapped__ = current  # type: ignore[attr-defined]
    work_graph_module._exclusive_anchor_keys = exclusive_anchor_keys


def _install_cpu_capacity_policy(scheduler_safety: Any) -> None:
    """Allow the host to expose all reviewed CPU/I/O capacity instead of a fixed cap."""
    current = scheduler_safety._cpu_capacity
    if getattr(current, "_mmm_host_configurable", False):
        return

    @wraps(current)
    def cpu_capacity() -> int:
        raw = os.environ.get("MMM_CPU_IO_WORKERS", "").strip()
        if not raw:
            return current()
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError("MMM_CPU_IO_WORKERS must be a positive integer") from exc
        if value < 1:
            raise ValueError("MMM_CPU_IO_WORKERS must be a positive integer")
        return value

    cpu_capacity._mmm_host_configurable = True  # type: ignore[attr-defined]
    cpu_capacity.__wrapped__ = current  # type: ignore[attr-defined]
    scheduler_safety._cpu_capacity = cpu_capacity


def _replace_stage_locks_with_anchor_fencing(work_graph_module: Any) -> None:
    """Replace global stage critical sections with WorkGraph collision edges."""
    from . import scheduler_parallel_safety_contract as scheduler_safety

    anchor_resolver = getattr(work_graph_module, "_exclusive_anchor_keys", None)
    if not callable(anchor_resolver) or not getattr(anchor_resolver, "_mmm_unscoped_fallback", False):
        return
    scheduler_safety._STAGE_WRITE_LOCKS.clear()
    scheduler_safety._SERIAL_CPU_STAGES = ()
    _install_cpu_capacity_policy(scheduler_safety)


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    with _INIT_LOCK:
        if _INSTALLED:
            return
        from . import custom_module_generator, project_index, work_graph

        _configure_pipeline_granularity()
        _install_custom_generator_lock(custom_module_generator)
        _install_project_index_parallel_scan(project_index)
        _install_project_index_snapshot_lock(project_index)
        _install_exact_anchor_fallback(work_graph)
        _replace_stage_locks_with_anchor_fencing(work_graph)
        _INSTALLED = True


__all__ = ["install"]
