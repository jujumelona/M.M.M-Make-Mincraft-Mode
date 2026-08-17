from __future__ import annotations

import re
from functools import wraps
from pathlib import PurePosixPath
from typing import Any, Iterable


_MARKER = "_mmm_source_snapshot_preconditions_v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourcePatchPreconditionError(ValueError):
    """Raised when a model patch cannot be bound to the host source snapshot."""


def _normalized_path(value: Any) -> str:
    return PurePosixPath(str(value or "").replace("\\", "/")).as_posix()


def _valid_sha256(value: Any) -> bool:
    return bool(_SHA256.fullmatch(str(value or "").strip().lower()))


def _snapshot_sha256(generator: Any, path: str) -> str:
    """Return the unique SHA committed by the generator's pre-decode ProjectIndex."""
    index = getattr(generator, "_cached_index", None)
    files = getattr(index, "files", ()) if index is not None else ()
    matches = {
        str(getattr(item, "sha256", "")).strip().lower()
        for item in files
        if _normalized_path(getattr(item, "path", "")) == path
        and _valid_sha256(getattr(item, "sha256", ""))
    }
    if len(matches) > 1:
        raise SourcePatchPreconditionError(
            f"ProjectIndex contains conflicting source identities for {path}."
        )
    return next(iter(matches), "")


def bind_source_snapshot_preconditions(
    generator: Any,
    operations: Iterable[dict[str, Any]],
) -> None:
    """Bind model mutations to the exact host snapshot used for generation.

    The model owns desired content, not optimistic-concurrency identities. For an
    existing indexed file, replace/edit operations inherit the pre-decode ProjectIndex
    SHA when the model omitted it. A create aimed at an already indexed file is safely
    normalized to replace. A model-supplied SHA that disagrees with the host snapshot
    is rejected before the transaction reaches the patcher.

    We deliberately never derive a missing precondition from the *current* live file:
    if source changed after the model observed it, TransactionalSourcePatcher must still
    detect that stale snapshot and refuse the write.
    """
    for raw in operations:
        if not isinstance(raw, dict):
            continue
        operation = str(raw.get("operation", "")).strip().lower()
        if operation not in {"create", "replace", "edit"}:
            continue
        path = _normalized_path(raw.get("path", ""))
        if not path or path == ".":
            continue
        snapshot_sha = _snapshot_sha256(generator, path)

        if operation == "create" and snapshot_sha:
            if not isinstance(raw.get("content"), str):
                raise SourcePatchPreconditionError(
                    f"Existing source {path} cannot be normalized from create without content."
                )
            raw["operation"] = "replace"
            raw["expected_sha256"] = snapshot_sha
            continue

        if operation not in {"replace", "edit"}:
            continue

        supplied = str(raw.get("expected_sha256", "")).strip().lower()
        if snapshot_sha:
            if supplied and _valid_sha256(supplied) and supplied != snapshot_sha:
                raise SourcePatchPreconditionError(
                    f"Model patch precondition disagrees with the observed source snapshot for {path}."
                )
            raw["expected_sha256"] = snapshot_sha
            continue

        if not _valid_sha256(supplied):
            raise SourcePatchPreconditionError(
                f"No observed source SHA is available to authorize {operation} for {path}."
            )


def install(custom_module_generator_module: Any) -> None:
    cls = custom_module_generator_module.CustomModuleGenerator
    current = cls._validate_operations
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def validate(self: Any, operations: list[dict[str, Any]]) -> None:
        bind_source_snapshot_preconditions(self, operations)
        return current(self, operations)

    setattr(validate, _MARKER, True)
    cls._validate_operations = validate


__all__ = [
    "SourcePatchPreconditionError",
    "bind_source_snapshot_preconditions",
    "install",
]
