from __future__ import annotations

"""Read the host-owned platform identity from one concrete project tree.

This module deliberately owns only local project evidence. Executable provider
resolution remains in :mod:`platform_catalog`, keeping low-level tool runtime code
independent from the platform registry and its planning/research dependency graph.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .gradle_properties import read_gradle_properties


@dataclass(frozen=True)
class ProjectPlatformIdentity:
    minecraft_version: str
    loader: str
    lock_values: Mapping[str, Any] | None
    gradle_properties: Mapping[str, str]


def _project_platform_lock(root: Path) -> Path | None:
    direct = root / ".minecraft_ai" / "platform-lock.json"
    if direct.is_file() and not direct.is_symlink():
        return direct
    if root.name != "project":
        return None
    checkpoint_root = root.parent
    checkpoint_directory = checkpoint_root.parent
    metadata_root = checkpoint_directory.parent
    key = checkpoint_root.name
    valid_key = len(key) == 64 and all(c in "0123456789abcdef" for c in key)
    if checkpoint_directory.name != ".mmm-custom-checkpoints" or metadata_root.name != ".minecraft_ai" or not valid_key:
        return None
    inherited = metadata_root / "platform-lock.json"
    return inherited if inherited.is_file() and not inherited.is_symlink() else None



def _fabric_descriptor_identifies_project(root: Path) -> bool:
    descriptor = root / "src" / "main" / "resources" / "fabric.mod.json"
    if not descriptor.is_file() or descriptor.is_symlink():
        return False
    try:
        raw = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(raw, dict):
        return False
    return (
        type(raw.get("schemaVersion")) is int
        and raw["schemaVersion"] >= 1
        and isinstance(raw.get("id"), str)
        and bool(raw["id"].strip())
        and isinstance(raw.get("version"), str)
        and bool(raw["version"].strip())
    )


def project_platform_identity(project_root: str | Path) -> ProjectPlatformIdentity:
    root = Path(project_root).expanduser().resolve()
    lock_file = _project_platform_lock(root)
    if lock_file is not None:
        raw = json.loads(lock_file.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Generated platform lock must be an object.")
        version = str(raw.get("minecraft_version") or "").strip()
        loader = str(raw.get("loader") or "").strip().casefold()
        if not version or not loader:
            raise ValueError("Generated platform lock must bind minecraft_version and loader.")
        return ProjectPlatformIdentity(version, loader, raw, {})

    properties = read_gradle_properties(root / "gradle.properties")
    version = properties.get("minecraft_version", "").strip()
    if not version:
        raise ValueError("Existing project minecraft_version is missing.")
    loader = properties.get("loader", "").strip().casefold()
    if not loader:
        fabric_properties = properties.get("loader_version") and properties.get("fabric_version")
        if fabric_properties or _fabric_descriptor_identifies_project(root):
            loader = "fabric"
        else:
            raise ValueError("Existing project loader could not be identified unambiguously.")
    return ProjectPlatformIdentity(version, loader, None, properties)


__all__ = ["ProjectPlatformIdentity", "project_platform_identity"]
