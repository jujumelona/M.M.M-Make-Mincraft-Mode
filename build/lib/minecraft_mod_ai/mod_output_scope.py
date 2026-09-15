from __future__ import annotations

from collections.abc import Iterable
from pathlib import PurePosixPath
from typing import Any


class ModOutputScopeError(ValueError):
    """Raised when a write targets a standalone Minecraft world artifact."""


_STANDALONE_SUFFIXES = frozenset(
    {
        ".schem",
        ".schematic",
        ".litematic",
        ".mcworld",
        ".mca",
        ".mcr",
    }
)
_STANDALONE_ROOT_FILES = frozenset(
    {
        "level.dat",
        "level.dat_old",
        "session.lock",
        "uid.dat",
    }
)
_STANDALONE_ROOT_DIRS = frozenset(
    {
        "region",
        "entities",
        "poi",
        "playerdata",
        "stats",
        "advancements",
        "dim-1",
        "dim1",
    }
)
_BUILDER_OUTPUT_NAMES = frozenset(
    {
        "buildspec.json",
        "buildspec.yaml",
        "buildspec.yml",
        "block_delta.json",
        "block-delta.json",
    }
)
_BUILDER_NPZ_NAMES = frozenset(
    {
        "blocks.npz",
        "block_delta.npz",
        "block-delta.npz",
        "block_deltas.npz",
        "block-deltas.npz",
        "world_delta.npz",
        "world-delta.npz",
    }
)
_BUILDER_NPZ_DIRS = frozenset(
    {
        "builder",
        "buildspec",
        "block_delta",
        "block-delta",
    }
)
_BUILDER_NPZ_MARKERS = (
    "block_delta",
    "block-delta",
    "blockdelta",
    "world_delta",
    "world-delta",
)
_WORLD_CONTAINER_DIRS = frozenset({"world", "worlds"})


def _normalized_parts(value: Any) -> tuple[str, ...]:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ()
    path = PurePosixPath(raw)
    return tuple(part.casefold() for part in path.parts if part not in {"", "."})


def _is_builder_npz(parts: tuple[str, ...], name: str, suffix: str) -> bool:
    if suffix != ".npz":
        return False
    return (
        name in _BUILDER_NPZ_NAMES
        or any(marker in name for marker in _BUILDER_NPZ_MARKERS)
        or any(part in _BUILDER_NPZ_DIRS for part in parts[:-1])
    )


def _world_save_violation(parts: tuple[str, ...], name: str) -> str | None:
    # A normal mod source tree can legitimately contain packages named ``world`` or
    # resource paths named ``advancements``. Only recognize save-layout directories
    # at a project/save root, or beneath an explicit saves/world container.
    if len(parts) == 1 and name in _STANDALONE_ROOT_FILES:
        return "Minecraft world-save root file"
    if parts[0] in _STANDALONE_ROOT_DIRS:
        return "Minecraft world-save root directory"
    if "saves" in parts:
        return "Minecraft saves directory"

    for index, part in enumerate(parts[:-1]):
        if part not in _WORLD_CONTAINER_DIRS:
            continue
        child = parts[index + 1]
        if child in _STANDALONE_ROOT_DIRS or child in _STANDALONE_ROOT_FILES:
            return "Minecraft world-save layout"
    return None


def mod_output_scope_violation(path: Any) -> str | None:
    """Return why *path* is not a mod-project output, otherwise ``None``.

    The policy intentionally does not use a broad extension allowlist: real Fabric
    projects may contain arbitrary Java/Kotlin sources, config, docs and datapack
    resources. It blocks only output classes owned by a standalone map/world builder.
    In particular, mod-native ``data/<namespace>/worldgen``, ``dimension`` and
    ``structure`` resources remain legal, including structure-template ``.nbt`` files.
    """

    parts = _normalized_parts(path)
    if not parts:
        return None
    name = parts[-1]
    suffix = PurePosixPath(name).suffix.casefold()

    if suffix in _STANDALONE_SUFFIXES:
        return f"standalone world/builder artifact suffix {suffix}"
    if name in _BUILDER_OUTPUT_NAMES or name.endswith(".buildspec"):
        return "Builder/BuildSpec output"
    if _is_builder_npz(parts, name, suffix):
        return "Builder NPZ block-delta output"
    return _world_save_violation(parts, name)


def validate_mod_output_path(path: Any) -> None:
    violation = mod_output_scope_violation(path)
    if violation is None:
        return
    raise ModOutputScopeError(
        f"Output path is outside M.M.M's mod-project scope: {path!s} ({violation}). "
        "Use mod-owned source or datapack/worldgen resources under the project source "
        "tree; standalone worlds, schematics and Builder block deltas are not writable."
    )


def validate_mod_patch_operations(operations: Iterable[dict[str, Any]]) -> None:
    """Validate every path exposed by a source-patch transaction."""

    for operation in operations:
        if not isinstance(operation, dict):
            continue
        path = operation.get("path")
        if isinstance(path, str) and path.strip():
            validate_mod_output_path(path)


__all__ = [
    "ModOutputScopeError",
    "mod_output_scope_violation",
    "validate_mod_output_path",
    "validate_mod_patch_operations",
]
