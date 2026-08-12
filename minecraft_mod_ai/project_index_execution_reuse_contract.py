from __future__ import annotations

from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

_POST_GENERATION: ContextVar[bool] = ContextVar(
    "mmm_project_index_post_generation",
    default=False,
)
_EXECUTION_INDEXES: ContextVar[dict[tuple[str, int], Any] | None] = ContextVar(
    "mmm_project_index_execution_cache",
    default=None,
)


def _root_key(project_root: str | Path) -> str:
    return str(Path(project_root).expanduser().resolve())


def _cache_key(project_root: str | Path, policy: Any | None) -> tuple[str, int]:
    return _root_key(project_root), 0 if policy is None else id(policy)


def _cached_indexes_for_root(
    cache: dict[tuple[str, int], Any],
    project_root: str | Path,
) -> tuple[Any, ...]:
    root = _root_key(project_root)
    seen: set[int] = set()
    indexes: list[Any] = []
    for (cached_root, _policy_id), index in cache.items():
        if cached_root != root:
            continue
        identity = id(index)
        if identity in seen:
            continue
        seen.add(identity)
        indexes.append(index)
    return tuple(indexes)


def _evict_root(
    cache: dict[tuple[str, int], Any],
    project_root: str | Path,
) -> None:
    root = _root_key(project_root)
    for key in tuple(cache):
        if key[0] == root:
            cache.pop(key, None)


def _receipt_paths(value: Any) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()

    def add(raw: Any) -> None:
        if not isinstance(raw, (str, Path)):
            return
        path = str(raw).strip()
        if path and path not in seen:
            seen.add(path)
            ordered.append(path)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if (
                node.get("operation") in {"create", "replace", "edit", "delete"}
                and isinstance(node.get("path"), str)
            ):
                add(node["path"])
            for key in (
                "touched_paths",
                "written_files",
                "deleted_files",
                "removed_files",
            ):
                paths = node.get(key)
                if isinstance(paths, (list, tuple, set)):
                    for path in paths:
                        add(path)
            files = node.get("files")
            if isinstance(files, (list, tuple, set)):
                for path in files:
                    add(path)
            for child in node.values():
                visit(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                visit(child)

    visit(value)
    return tuple(ordered)


def install(orchestrator_module: Any) -> None:
    """Reuse one authoritative post-generation ProjectIndex inside an execution.

    Generation itself keeps its existing fresh shared index. Only after generation
    completes do we cache the next ProjectIndex construction. Resource hardening then
    updates every cached index for the same project root from its transactional
    receipts, independent of which derived tuning policy produced those receipts. If a
    mutating receipt cannot prove which files changed, all cached indexes for that root
    are discarded and the next construction falls back to the original full scan.
    """

    cls = orchestrator_module.CompleteProductionOrchestrator

    current_execute = cls.execute
    if not getattr(current_execute, "_mmm_project_index_execution_scope", False):

        @wraps(current_execute)
        def execute(self: Any, *args: Any, **kwargs: Any):
            index_token = _EXECUTION_INDEXES.set({})
            phase_token = _POST_GENERATION.set(False)
            try:
                return current_execute(self, *args, **kwargs)
            finally:
                _POST_GENERATION.reset(phase_token)
                _EXECUTION_INDEXES.reset(index_token)

        execute._mmm_project_index_execution_scope = True  # type: ignore[attr-defined]
        cls.execute = execute

    current_generation = cls._execute_generation_work
    if not getattr(current_generation, "_mmm_marks_post_generation", False):

        @wraps(current_generation)
        def execute_generation_work(self: Any, *args: Any, **kwargs: Any):
            result = current_generation(self, *args, **kwargs)
            _POST_GENERATION.set(True)
            return result

        execute_generation_work._mmm_marks_post_generation = True  # type: ignore[attr-defined]
        cls._execute_generation_work = execute_generation_work

    current_project_index = orchestrator_module.ProjectIndex
    if not getattr(current_project_index, "_mmm_post_generation_reuse", False):

        def project_index(project_root: str | Path, *args: Any, **kwargs: Any):
            if not _POST_GENERATION.get():
                return current_project_index(project_root, *args, **kwargs)
            cache = _EXECUTION_INDEXES.get()
            if cache is None:
                return current_project_index(project_root, *args, **kwargs)
            policy = kwargs.get("policy")
            key = _cache_key(project_root, policy)
            cached = cache.get(key)
            if cached is not None:
                return cached
            created = current_project_index(project_root, *args, **kwargs)
            cache[key] = created
            return created

        project_index._mmm_post_generation_reuse = True  # type: ignore[attr-defined]
        project_index.__wrapped__ = current_project_index  # type: ignore[attr-defined]
        orchestrator_module.ProjectIndex = project_index

    current_tune = orchestrator_module.tune_gradle_resources
    if not getattr(current_tune, "_mmm_updates_execution_project_index", False):

        @wraps(current_tune)
        def tune_gradle_resources(project_root: str | Path, *args: Any, **kwargs: Any):
            receipt = current_tune(project_root, *args, **kwargs)
            if not _POST_GENERATION.get():
                return receipt
            cache = _EXECUTION_INDEXES.get()
            if cache is None:
                return receipt
            indexes = _cached_indexes_for_root(cache, project_root)
            if not indexes:
                return receipt
            paths = _receipt_paths(receipt)
            status = str(receipt.get("status", "")) if isinstance(receipt, dict) else ""
            if paths:
                for index in indexes:
                    index.update_files(paths)
            elif status not in {"", "UNCHANGED"}:
                # Unknown mutating receipt shape: fail safe to a fresh scan on the
                # next ProjectIndex construction rather than trusting stale state.
                _evict_root(cache, project_root)
            return receipt

        tune_gradle_resources._mmm_updates_execution_project_index = True  # type: ignore[attr-defined]
        orchestrator_module.tune_gradle_resources = tune_gradle_resources


__all__ = [
    "_cached_indexes_for_root",
    "_receipt_paths",
    "install",
]
