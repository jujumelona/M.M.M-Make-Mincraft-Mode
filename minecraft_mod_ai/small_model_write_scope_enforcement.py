from __future__ import annotations

"""Fail-closed write-scope enforcement for small-model custom generation.

Planning owns the exact target files. This layer carries that ownership through the
staged-workspace transaction boundary so a coder cannot broaden a task merely because
a path lives under a generally mutable source directory.
"""

import threading
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from functools import wraps
from pathlib import PurePosixPath
from typing import Any

from .implementation_template_contract import build_implementation_template

_SCOPE = threading.local()
_INSTALLED = False


def _normalize_path(value: Any) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    normalized = candidate.as_posix()
    return "" if normalized in {"", "."} else normalized


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


def install(*, custom_module_generator_module: Any, host_grounding_module: Any) -> None:
    """Compose exact task ownership into the live custom-coder transaction path."""

    global _INSTALLED
    if _INSTALLED:
        return

    generator_type = custom_module_generator_module.CustomModuleGenerator
    original_generate = generator_type.generate
    original_validate = generator_type._validate_operations

    @wraps(original_generate)
    def scoped_generate(
        self: Any,
        project_root: Any,
        *,
        module: Any,
        research_modules: Any = (),
        minecraft_version: str | None = None,
        loader: str | None = None,
        mappings: str | None = None,
    ) -> Any:
        writable_paths = exact_task_writable_paths(module)
        with _active_write_scope(writable_paths):
            return original_generate(
                self,
                project_root,
                module=module,
                research_modules=research_modules,
                minecraft_version=minecraft_version,
                loader=loader,
                mappings=mappings,
            )

    @wraps(original_validate)
    def exact_validate(self: Any, operations: list[dict[str, Any]]) -> None:
        original_validate(self, operations)
        writable_paths = getattr(_SCOPE, "allowed_paths", None)
        if writable_paths is None:
            return
        validate_exact_task_operations(
            operations,
            writable_paths,
            error_type=custom_module_generator_module.CustomModuleGenerationError,
        )

    scoped_generate._mmm_exact_task_write_scope = True  # type: ignore[attr-defined]
    exact_validate._mmm_exact_task_write_scope = True  # type: ignore[attr-defined]
    generator_type.generate = scoped_generate
    generator_type._validate_operations = exact_validate

    # Keep staged-operation discovery aligned with the same coarse custom-coder policy
    # that is published to the model. Exact task ownership is enforced above it.
    custom_module_generator_module._agent_mutable_path = (
        host_grounding_module.custom_module_path_allowed
    )
    _INSTALLED = True


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
    "exact_task_writable_paths",
    "install",
    "validate_exact_task_operations",
]
