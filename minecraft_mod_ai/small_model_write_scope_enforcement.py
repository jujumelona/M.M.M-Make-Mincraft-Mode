from __future__ import annotations

"""Fail-closed write-scope enforcement for evidence-owned and authored-design tasks.

Planning owns exact target files for evidence-backed production tasks. Saved authored
designs instead receive a host-owned bounded-root authority. This existing generation
wrapper owns the request-scoped authority lifetime so downstream tool and staged-patch
guards consume one canonical value without adding another runtime monkeypatch layer.
"""

import threading
from collections.abc import Mapping, Sequence
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

from .direct_task_mutation_authority_contract import (
    _CURRENT_AUTHORITY,
    compile_direct_task_mutation_authority,
)
from .implementation_template_contract import build_implementation_template
from .mutation_authority import CURRENT_MUTATION_AUTHORITY

_SCOPE = threading.local()


def _normalize_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    normalized = candidate.as_posix()
    return "" if normalized in {"", "."} else normalized


def _declares_evidence_task(module: Any) -> bool:
    config = getattr(module, "config", None)
    return isinstance(config, Mapping) and "evidence_task" in config


def exact_task_writable_paths(module: Any) -> tuple[str, ...]:
    """Return the host-authored exact writable file set for one production module."""

    config = getattr(module, "config", None)
    if not isinstance(config, Mapping):
        raise RuntimeError("TASK_WRITE_SCOPE_REQUIRED: module config is unavailable")
    task = config.get("evidence_task")
    if not isinstance(task, Mapping):
        raise RuntimeError("TASK_WRITE_SCOPE_REQUIRED: config.evidence_task is unavailable")

    contract = build_implementation_template(task)
    boundaries = contract.get("protected_boundaries")
    if not isinstance(boundaries, Mapping):
        raise RuntimeError("TASK_WRITE_SCOPE_REQUIRED: coder contract has no protected_boundaries")
    raw_paths = boundaries.get("writable_paths")
    if not isinstance(raw_paths, Sequence) or isinstance(raw_paths, (str, bytes, bytearray)):
        raise RuntimeError("TASK_WRITE_SCOPE_REQUIRED: coder contract has no writable_paths")

    paths: list[str] = []
    for value in raw_paths:
        normalized = _normalize_path(value)
        if not normalized:
            raise RuntimeError("TASK_WRITE_SCOPE_INVALID: writable path is not project-relative")
        if normalized not in paths:
            paths.append(normalized)
    if not paths:
        raise RuntimeError("TASK_WRITE_SCOPE_REQUIRED: exact writable path set is empty")
    return tuple(paths)


def validate_exact_task_operations(
    operations: Sequence[Mapping[str, Any]],
    allowed_paths: Sequence[str],
    *,
    error_type: type[Exception] = RuntimeError,
) -> None:
    """Reject any staged mutation not owned by the current task contract."""

    allowed = frozenset(_normalize_path(path) for path in allowed_paths)
    if "" in allowed or not allowed:
        raise error_type("TASK_WRITE_SCOPE_REQUIRED: exact writable path set is invalid")
    for operation in operations:
        path = _normalize_path(operation.get("path"))
        if not path or path not in allowed:
            raise error_type(
                "TASK_WRITE_SCOPE_ESCAPE: staged operation is outside exact writable_paths: "
                f"{path or '<invalid>'}"
            )


@contextmanager
def _active_write_scope(paths: Sequence[str]):
    previous = getattr(_SCOPE, "allowed_paths", None)
    _SCOPE.allowed_paths = tuple(paths)
    try:
        yield
    finally:
        if previous is None:
            try:
                del _SCOPE.allowed_paths
            except AttributeError:
                pass
        else:
            _SCOPE.allowed_paths = previous


@contextmanager
def _active_generation_authority(module: Any):
    """Bind one host authority to every write guard for exactly one generation call."""

    authority = compile_direct_task_mutation_authority(module)
    envelope_token = _CURRENT_AUTHORITY.set(authority)
    mutation_token = CURRENT_MUTATION_AUTHORITY.set(
        authority.mutation_authority if authority is not None else None
    )
    try:
        with ExitStack() as stack:
            if _declares_evidence_task(module):
                stack.enter_context(_active_write_scope(exact_task_writable_paths(module)))
            yield
    finally:
        CURRENT_MUTATION_AUTHORITY.reset(mutation_token)
        _CURRENT_AUTHORITY.reset(envelope_token)


def generation_authority_scoped(func: Any) -> Any:
    """Bind host mutation authority directly at the reviewed generation definition."""

    @wraps(func)
    def scoped_generate(
        self: Any,
        project_root: Any,
        *,
        module: Any,
        research_modules: Any = (),
        minecraft_version: str | None = None,
        loader: str | None = None,
        mappings: str | None = None,
        execution_feedback: Mapping[str, Any] | None = None,
    ) -> Any:
        with _active_generation_authority(module):
            return func(
                self,
                project_root,
                module=module,
                research_modules=research_modules,
                minecraft_version=minecraft_version,
                loader=loader,
                mappings=mappings,
                execution_feedback=execution_feedback,
            )

    scoped_generate._mmm_exact_task_write_scope = True  # type: ignore[attr-defined]
    return scoped_generate


def exact_task_operation_validator(
    error_type: type[Exception],
):
    """Attach exact task-path validation at the source-owned validator definition."""

    def decorate(func: Any) -> Any:
        @wraps(func)
        def exact_validate(self: Any, operations: list[dict[str, Any]]) -> None:
            func(self, operations)
            writable_paths = getattr(_SCOPE, "allowed_paths", None)
            if writable_paths is None:
                return
            validate_exact_task_operations(
                operations,
                writable_paths,
                error_type=error_type,
            )

        exact_validate._mmm_exact_task_write_scope = True  # type: ignore[attr-defined]
        return exact_validate

    return decorate


def assert_installed(*, custom_module_generator_module: Any, host_grounding_module: Any) -> None:
    generator_type = custom_module_generator_module.CustomModuleGenerator
    if not getattr(generator_type.generate, "_mmm_exact_task_write_scope", False):
        raise RuntimeError("exact task write-scope generate wrapper is not installed")
    if not getattr(generator_type._validate_operations, "_mmm_exact_task_write_scope", False):
        raise RuntimeError("exact task write-scope validator wrapper is not installed")
    if custom_module_generator_module._agent_mutable_path is not host_grounding_module.custom_module_path_allowed:
        raise RuntimeError("custom coder coarse write policy is not host-grounding owned")


__all__ = [
    "assert_installed",
    "exact_task_operation_validator",
    "exact_task_writable_paths",
    "generation_authority_scoped",
    "validate_exact_task_operations",
]