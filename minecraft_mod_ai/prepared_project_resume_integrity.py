from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .platform_catalog import adapter_for_lock_values
from .project_edit import ProjectEditError, inspect_fabric_project
from .spec import Proposal
from .toolchain_contract import fabric_dependency_predicates


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _entrypoint_values(value: Any) -> set[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return set()
    result: set[str] = set()
    for item in value:
        if isinstance(item, str) and item.strip():
            result.add(item.strip())
        elif isinstance(item, Mapping):
            raw = item.get("value")
            if isinstance(raw, str) and raw.strip():
                result.add(raw.strip())
    return result


def _json_object(path: Path) -> dict[str, Any] | None:
    if not _regular_file(path):
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _gradle_properties(path: Path) -> dict[str, str] | None:
    if not _regular_file(path):
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    except (OSError, UnicodeError):
        return None
    result: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def prepared_project_matches_spec(project_root: Path, spec: Any) -> bool:
    """Prove that a resumable base workspace still matches the approved platform lock."""
    root = Path(project_root).expanduser()
    if not root.is_dir() or root.is_symlink():
        return False
    try:
        root = root.resolve(strict=True)
        info = inspect_fabric_project(root)
        adapter = adapter_for_lock_values(spec.platform)
        expected_depends = fabric_dependency_predicates(spec.platform)
    except (OSError, RuntimeError, ValueError, ProjectEditError):
        return False

    build_files = (root / "build.gradle", root / "build.gradle.kts")
    if not any(_regular_file(path) for path in build_files):
        return False
    if not _regular_file(info.main_java):
        return False
    if info.mod_id != spec.mod_id or info.package_name != spec.package_name:
        return False

    resource_root = root / "src/main/resources"
    fabric_path = resource_root / "fabric.mod.json"
    pack_path = resource_root / "pack.mcmeta"
    lang_root = resource_root / "assets" / spec.mod_id / "lang"
    en_path = lang_root / "en_us.json"
    ko_path = lang_root / "ko_kr.json"
    properties_path = root / "gradle.properties"

    fabric = _json_object(fabric_path)
    pack = _json_object(pack_path)
    english = _json_object(en_path)
    korean = _json_object(ko_path)
    properties = _gradle_properties(properties_path)
    if any(value is None for value in (fabric, pack, english, korean, properties)):
        return False
    assert fabric is not None and pack is not None and properties is not None

    if fabric.get("id") != spec.mod_id or fabric.get("version") != "${version}" or fabric.get("environment") != "*":
        return False
    depends = fabric.get("depends")
    if not isinstance(depends, Mapping):
        return False
    if any(depends.get(name) != constraint for name, constraint in expected_depends.items()):
        return False

    main_class = "".join(part.capitalize() for part in spec.mod_id.split("_")) + "Mod"
    main_entrypoint = f"{spec.package_name}.{main_class}"
    gametest_entrypoint = main_entrypoint + "GameTests"
    entrypoints = fabric.get("entrypoints")
    if not isinstance(entrypoints, Mapping):
        return False
    if main_entrypoint not in _entrypoint_values(entrypoints.get("main")):
        return False
    if gametest_entrypoint not in _entrypoint_values(entrypoints.get("fabric-gametest")):
        return False

    pack_section = pack.get("pack")
    if not isinstance(pack_section, Mapping):
        return False
    if pack_section.get("pack_format") != adapter.resource_pack_format:
        return False

    expected_properties = {
        "minecraft_version": str(spec.platform.minecraft_version),
        "yarn_mappings": str(spec.platform.yarn_mappings),
        "loader_version": str(spec.platform.fabric_loader),
        "loom_version": str(spec.platform.fabric_loom),
        "fabric_version": str(spec.platform.fabric_api),
        "java_version": str(spec.platform.java_version),
    }
    if any(properties.get(key) != value for key, value in expected_properties.items()):
        return False
    return True


def prepared_project_cache_valid(project_root: Path) -> bool:
    """Validate a cached prepare-project checkpoint against its persisted base proposal."""
    root = Path(project_root).expanduser()
    proposal_path = root / ".minecraft_ai/base-proposal.json"
    payload = _json_object(proposal_path)
    if payload is None:
        return False
    try:
        proposal = Proposal.from_dict(payload)
    except (TypeError, ValueError):
        return False
    return prepared_project_matches_spec(root, proposal.spec)


__all__ = ["prepared_project_cache_valid", "prepared_project_matches_spec"]
