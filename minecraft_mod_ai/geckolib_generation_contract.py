from __future__ import annotations

"""Pure deterministic preflight for GeckoLib generation targets."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .platform_catalog import PlatformAdapter, adapter_from_project
from .project_edit import FabricProjectInfo, ProjectEditError, inspect_fabric_project

_DEPENDENCIES_BLOCK = re.compile(r"\bdependencies\s*\{")
_GECKOLIB_DEPENDENCY_MARKER = "// MMM:geckolib:dependency"


class GeckoLibGenerationContractError(ValueError):
    """The approved project cannot safely enter GeckoLib file generation."""


@dataclass(frozen=True)
class GeckoLibGenerationTarget:
    info: FabricProjectInfo
    adapter: PlatformAdapter


def _read_utf8(path: Path, *, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise GeckoLibGenerationContractError(
            f"{label} must be readable UTF-8 text: {path}"
        ) from exc


def validate_geckolib_project_preflight(
    info: FabricProjectInfo,
) -> PlatformAdapter:
    """Validate project state that existing GeckoLib code would otherwise reject late."""

    try:
        adapter = adapter_from_project(info.root)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise GeckoLibGenerationContractError(
            f"GeckoLib requires an executable platform target before generation: {exc}"
        ) from exc

    metadata_text = _read_utf8(info.fabric_mod_json, label="fabric.mod.json")
    try:
        metadata = json.loads(metadata_text)
    except json.JSONDecodeError as exc:
        raise GeckoLibGenerationContractError(
            "fabric.mod.json must be valid JSON before GeckoLib generation."
        ) from exc
    if not isinstance(metadata, dict):
        raise GeckoLibGenerationContractError(
            "fabric.mod.json must be an object before GeckoLib generation."
        )

    depends = metadata.get("depends")
    if depends is not None and not isinstance(depends, dict):
        raise GeckoLibGenerationContractError(
            "fabric.mod.json depends must be an object."
        )
    entrypoints = metadata.get("entrypoints")
    if not isinstance(entrypoints, dict):
        raise GeckoLibGenerationContractError(
            "fabric.mod.json entrypoints must be an object."
        )
    client = entrypoints.get("client")
    if client is not None and not isinstance(client, list):
        raise GeckoLibGenerationContractError(
            "fabric.mod.json entrypoints.client must be a list."
        )

    build_file = info.root / "build.gradle"
    if not build_file.is_file() or build_file.is_symlink():
        raise GeckoLibGenerationContractError(
            "build.gradle is required before GeckoLib dependency generation."
        )
    build_text = _read_utf8(build_file, label="build.gradle")
    if (
        _GECKOLIB_DEPENDENCY_MARKER not in build_text
        and _DEPENDENCIES_BLOCK.search(build_text) is None
    ):
        raise GeckoLibGenerationContractError(
            "build.gradle has no dependencies block for GeckoLib insertion."
        )

    if info.main_java.is_file() and not info.main_java.is_symlink():
        _read_utf8(info.main_java, label="Fabric main entrypoint")

    return adapter


def preflight_geckolib_generation_target(
    project_root: str | Path,
    *,
    mod_id: str,
    package_name: str,
) -> GeckoLibGenerationTarget:
    """Inspect and validate every project condition knowable before writes begin."""

    try:
        info = inspect_fabric_project(project_root)
    except (ProjectEditError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeckoLibGenerationContractError(
            f"GeckoLib project inspection failed before generation: {exc}"
        ) from exc

    if info.mod_id != mod_id or info.package_name != package_name:
        raise GeckoLibGenerationContractError(
            "GeckoLib target does not match fabric.mod.json."
        )

    adapter = validate_geckolib_project_preflight(info)
    return GeckoLibGenerationTarget(info=info, adapter=adapter)


__all__ = [
    "GeckoLibGenerationContractError",
    "GeckoLibGenerationTarget",
    "preflight_geckolib_generation_target",
    "validate_geckolib_project_preflight",
]
