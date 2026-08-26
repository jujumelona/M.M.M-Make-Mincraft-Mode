from __future__ import annotations

"""Explicit execution-scoped ProjectIndex reuse.

The orchestrator calls these helpers directly; this module never replaces imported
classes or functions at runtime.
"""

from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])
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


def _evict_root(cache: dict[tuple[str, int], Any], project_root: str | Path) -> None:
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


def execution_scoped(function: _F) -> _F:
    """Give one orchestrator execution an isolated post-generation index cache."""

    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any):
        index_token = _EXECUTION_INDEXES.set({})
        phase_token = _POST_GENERATION.set(False)
        try:
            return function(*args, **kwargs)
        finally:
            _POST_GENERATION.reset(phase_token)
            _EXECUTION_INDEXES.reset(index_token)

    wrapped._mmm_project_index_execution_scope = True  # type: ignore[attr-defined]
    return wrapped  # type: ignore[return-value]


def mark_post_generation() -> None:
    _POST_GENERATION.set(True)


def project_index(
    factory: Callable[..., Any],
    project_root: str | Path,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Construct normally before generation; reuse by root/policy afterwards."""
    if not _POST_GENERATION.get():
        return factory(project_root, *args, **kwargs)
    cache = _EXECUTION_INDEXES.get()
    if cache is None:
        return factory(project_root, *args, **kwargs)
    policy = kwargs.get("policy")
    key = _cache_key(project_root, policy)
    cached = cache.get(key)
    if cached is not None:
        return cached
    created = factory(project_root, *args, **kwargs)
    cache[key] = created
    return created


def _update_from_receipt(project_root: str | Path, receipt: Any) -> None:
    if not _POST_GENERATION.get():
        return
    cache = _EXECUTION_INDEXES.get()
    if cache is None:
        return
    indexes = _cached_indexes_for_root(cache, project_root)
    if not indexes:
        return
    paths = _receipt_paths(receipt)
    status = str(receipt.get("status", "")) if isinstance(receipt, dict) else ""
    if paths:
        for index in indexes:
            index.update_files(paths)
    elif status not in {"", "UNCHANGED"}:
        _evict_root(cache, project_root)


def tune_gradle_resources(project_root: str | Path, *args: Any, **kwargs: Any):
    """Run canonical resource tuning and reconcile active indexes from its receipt."""
    from .resource_tuning import tune_gradle_resources as tune

    receipt = tune(project_root, *args, **kwargs)
    _update_from_receipt(project_root, receipt)
    return receipt


def install(orchestrator_module: Any) -> None:
    """Compatibility verifier for callers that still invoke the former installer."""
    execute = orchestrator_module.CompleteProductionOrchestrator.execute
    if not getattr(execute, "_mmm_project_index_execution_scope", False):
        raise RuntimeError("ProjectIndex execution reuse must be owned by the orchestrator.")


__all__ = [
    "_cached_indexes_for_root",
    "_receipt_paths",
    "execution_scoped",
    "install",
    "mark_post_generation",
    "project_index",
    "tune_gradle_resources",
]
