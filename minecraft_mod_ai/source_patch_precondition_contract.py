from __future__ import annotations

import re
from functools import wraps
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


_MARKER = "_mmm_source_snapshot_preconditions_v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class SourcePatchPreconditionError(ValueError):
    """Raised when a model patch cannot be bound to the host source snapshot."""


def _normalized_path(value: Any) -> str:
    return PurePosixPath(str(value or "").replace("\\", "/")).as_posix()


def _valid_sha256(value: Any) -> bool:
    return bool(_SHA256.fullmatch(str(value or "").strip().lower()))


def _has_bound_snapshot(generator: Any) -> bool:
    return (
        getattr(generator, "_cached_index", None) is not None
        and getattr(generator, "_cached_root", None) is not None
    )


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

    We deliberately never derive a missing precondition from the current file. If a
    path was not in the exact-source ProjectIndex, the model is not allowed to invent a
    replace/edit identity for it. If source changes after the model observed it, the
    patcher still detects the stale snapshot and refuses the write.
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

        if not snapshot_sha:
            raise SourcePatchPreconditionError(
                f"No observed source SHA is available to authorize {operation} for {path}."
            )
        supplied = str(raw.get("expected_sha256", "")).strip().lower()
        if supplied and _valid_sha256(supplied) and supplied != snapshot_sha:
            raise SourcePatchPreconditionError(
                f"Model patch precondition disagrees with the observed source snapshot for {path}."
            )
        raw["expected_sha256"] = snapshot_sha


def preflight_source_patch_operations(
    generator: Any,
    operations: Iterable[dict[str, Any]],
) -> None:
    """Run every deterministic patcher check without mutating the staging tree."""
    from .source_patch import (
        SourcePatchError,
        TransactionalSourcePatcher,
        sha256_bytes,
    )

    root_value = getattr(generator, "_cached_root", None)
    if root_value is None:
        raise SourcePatchPreconditionError(
            "Custom patch validation has no bound project snapshot root."
        )
    root = Path(root_value).expanduser().resolve()
    try:
        patcher = TransactionalSourcePatcher(root)
        normalized = [patcher._normalize(raw) for raw in operations]
        if not normalized:
            raise SourcePatchError("At least one patch operation is required.")
        paths = [item["path"] for item in normalized]
        if len(paths) != len(set(paths)):
            raise SourcePatchError("A patch transaction may touch each path only once.")

        for item in normalized:
            target = patcher._path(
                item["path"],
                allow_missing=item["operation"] == "create",
            )
            exists = target.exists()
            if exists and (not target.is_file() or target.is_symlink()):
                raise SourcePatchError(
                    f"Patch target is not a regular file: {item['path']}"
                )
            before = target.read_bytes() if exists else None
            if item["operation"] == "create":
                if exists:
                    raise SourcePatchError(
                        f"Create target already exists: {item['path']}"
                    )
                continue

            if not exists:
                raise SourcePatchError(
                    f"Patch target does not exist: {item['path']}"
                )
            actual = sha256_bytes(before or b"")
            if item["expected_sha256"] != actual:
                raise SourcePatchError(
                    f"SHA-256 precondition failed for {item['path']}: "
                    f"{actual} != {item['expected_sha256']}"
                )
            if item["operation"] == "replace":
                after = item["content"].encode("utf-8")
            elif item["operation"] == "edit":
                after = patcher._edit(before or b"", item).encode("utf-8")
            else:
                continue
            if before == after or (
                before is not None
                and after is not None
                and before.replace(b"\r\n", b"\n") == after.replace(b"\r\n", b"\n")
            ):
                raise SourcePatchError(
                    f"Patch operation makes no change: {item['path']}"
                )
    except SourcePatchError as exc:
        raise SourcePatchPreconditionError(str(exc)) from exc


def install(custom_module_generator_module: Any) -> None:
    cls = custom_module_generator_module.CustomModuleGenerator
    current = cls._validate_operations
    if getattr(current, _MARKER, False):
        return

    @wraps(current)
    def validate(self: Any, operations: list[dict[str, Any]]) -> None:
        # Scope/type policy remains a usable standalone validator. Snapshot-bound
        # preconditions are a production-generation concern and activate only after
        # the generator has built the exact ProjectIndex it showed to the model.
        current(self, operations)
        if not _has_bound_snapshot(self):
            return
        bind_source_snapshot_preconditions(self, operations)
        preflight_source_patch_operations(self, operations)

    setattr(validate, _MARKER, True)
    cls._validate_operations = validate


__all__ = [
    "SourcePatchPreconditionError",
    "bind_source_snapshot_preconditions",
    "install",
    "preflight_source_patch_operations",
]
