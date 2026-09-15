from __future__ import annotations

"""Pure deterministic preflight for GeckoLib generation targets."""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .platform_catalog import PlatformAdapter, adapter_from_project
from .project_edit import FabricProjectInfo, ProjectEditError, inspect_fabric_project
from .scale_policy import ScalePolicy

_DEPENDENCIES_BLOCK = re.compile(r"\bdependencies\s*\{")
_GECKOLIB_DEPENDENCY_MARKER = "// MMM:geckolib:dependency"
_JAVA_TYPE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
_ARCHETYPES = frozenset(
    {"biped", "quadruped", "flying", "serpentine", "construct", "custom"}
)
_BEHAVIORS = frozenset({"hostile_melee", "neutral_melee", "passive", "npc"})
_SPAWN_GROUPS = frozenset(
    {"monster", "creature", "ambient", "water_creature", "misc"}
)


class GeckoLibGenerationContractError(ValueError):
    """The approved project cannot safely enter GeckoLib file generation."""


@dataclass(frozen=True)
class GeckoLibGenerationTarget:
    info: FabricProjectInfo
    adapter: PlatformAdapter


@dataclass(frozen=True)
class GeckoLibEntityGenerationInputs:
    """Canonical entity arguments shared by proposal preflight and generation."""

    texture_width: int
    texture_height: int
    max_health: float
    attack_damage: float
    movement_speed: float
    follow_range: float
    archetype: str
    behavior: str
    entity_width: float
    entity_height: float
    spawn_group: str
    custom_bones: list[dict[str, Any]] | None

    def generator_kwargs(self) -> dict[str, Any]:
        return {
            "texture_width": self.texture_width,
            "texture_height": self.texture_height,
            "max_health": self.max_health,
            "attack_damage": self.attack_damage,
            "movement_speed": self.movement_speed,
            "follow_range": self.follow_range,
            "archetype": self.archetype,
            "behavior": self.behavior,
            "entity_width": self.entity_width,
            "entity_height": self.entity_height,
            "spawn_group": self.spawn_group,
            "custom_bones": self.custom_bones,
        }


def validate_geckolib_entity_inputs(
    *,
    texture_width: int,
    texture_height: int,
    max_health: int | float,
    attack_damage: int | float,
    movement_speed: int | float,
    follow_range: int | float,
    archetype: str,
    behavior: str,
    entity_width: int | float,
    entity_height: int | float,
    spawn_group: str | None,
    custom_bones: list[dict[str, Any]] | None,
    policy: ScalePolicy | None = None,
) -> GeckoLibEntityGenerationInputs:
    """Validate the exact deterministic inputs consumed by the GeckoLib generator."""

    policy = policy or ScalePolicy.from_environment()
    policy.validate()
    if (
        type(texture_width) is not int
        or type(texture_height) is not int
        or not 1 <= texture_width <= policy.max_texture_dimension
        or not 1 <= texture_height <= policy.max_texture_dimension
    ):
        raise GeckoLibGenerationContractError(
            "Texture dimensions exceed configured resource policy."
        )

    numeric = {
        "max_health": max_health,
        "attack_damage": attack_damage,
        "movement_speed": movement_speed,
        "follow_range": follow_range,
        "entity_width": entity_width,
        "entity_height": entity_height,
    }
    normalized_numeric: dict[str, float] = {}
    for name, value in numeric.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0
        ):
            raise GeckoLibGenerationContractError(
                f"{name} must be a positive finite number."
            )
        normalized_numeric[name] = float(value)

    if archetype not in _ARCHETYPES or (
        archetype == "custom" and not custom_bones
    ):
        raise GeckoLibGenerationContractError(
            "Unknown or incomplete entity archetype."
        )
    if behavior not in _BEHAVIORS:
        raise GeckoLibGenerationContractError("Unknown entity behavior profile.")
    effective_spawn_group = spawn_group or (
        "monster" if behavior == "hostile_melee" else "creature"
    )
    if effective_spawn_group not in _SPAWN_GROUPS:
        raise GeckoLibGenerationContractError("Unknown spawn group.")

    return GeckoLibEntityGenerationInputs(
        texture_width=texture_width,
        texture_height=texture_height,
        max_health=normalized_numeric["max_health"],
        attack_damage=normalized_numeric["attack_damage"],
        movement_speed=normalized_numeric["movement_speed"],
        follow_range=normalized_numeric["follow_range"],
        archetype=archetype,
        behavior=behavior,
        entity_width=normalized_numeric["entity_width"],
        entity_height=normalized_numeric["entity_height"],
        spawn_group=effective_spawn_group,
        custom_bones=custom_bones,
    )


def geckolib_entity_inputs_from_module_config(
    kind: str,
    config: Mapping[str, Any],
    *,
    policy: ScalePolicy | None = None,
) -> GeckoLibEntityGenerationInputs:
    """Apply the orchestrator's coercion semantics before any generation can start."""

    behavior_default = "npc" if kind == "npc" else "hostile_melee"
    raw_custom_bones = config.get("custom_bones")
    custom_bones = raw_custom_bones if isinstance(raw_custom_bones, list) else None
    try:
        texture_width = int(config.get("texture_width", 64))
        texture_height = int(config.get("texture_height", 64))
        max_health = float(config.get("max_health", 80.0))
        attack_damage = float(config.get("attack_damage", 8.0))
        movement_speed = float(config.get("movement_speed", 0.27))
        follow_range = float(config.get("follow_range", 40.0))
        entity_width = float(config.get("entity_width", 0.8))
        entity_height = float(config.get("entity_height", 2.0))
    except (TypeError, ValueError, OverflowError) as exc:
        raise GeckoLibGenerationContractError(
            "Entity generation numeric fields cannot be converted to generator inputs."
        ) from exc

    spawn_group = (
        str(config["spawn_group"]) if config.get("spawn_group") else None
    )
    return validate_geckolib_entity_inputs(
        texture_width=texture_width,
        texture_height=texture_height,
        max_health=max_health,
        attack_damage=attack_damage,
        movement_speed=movement_speed,
        follow_range=follow_range,
        archetype=str(config.get("archetype", "biped")),
        behavior=str(config.get("behavior", behavior_default)),
        entity_width=entity_width,
        entity_height=entity_height,
        spawn_group=spawn_group,
        custom_bones=custom_bones,
        policy=policy,
    )


def validate_existing_geckolib_records(project_root: str | Path) -> None:
    """Validate persisted entity metadata reused while generating a new entity."""

    root = Path(project_root).expanduser().resolve()
    manifest = root / ".minecraft_ai/geckolib-entities.json"
    if manifest.exists() and (not manifest.is_file() or manifest.is_symlink()):
        raise GeckoLibGenerationContractError(
            "Existing GeckoLib entity index must be a regular file."
        )

    from .geckolib_generator import iter_geckolib_entity_records

    try:
        records = tuple(iter_geckolib_entity_records(root))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GeckoLibGenerationContractError(
            f"Existing GeckoLib entity records are invalid: {exc}"
        ) from exc

    for record in records:
        for field in ("class_name", "entity_class"):
            value = record.get(field)
            if not isinstance(value, str) or not _JAVA_TYPE.fullmatch(value):
                raise GeckoLibGenerationContractError(
                    f"Existing GeckoLib entity record {field} is not a Java type identifier."
                )
        for field in (
            "max_health",
            "attack_damage",
            "movement_speed",
            "follow_range",
            "entity_width",
            "entity_height",
        ):
            try:
                value = float(record.get(field))
            except (TypeError, ValueError, OverflowError) as exc:
                raise GeckoLibGenerationContractError(
                    f"Existing GeckoLib entity record {field} is not numeric."
                ) from exc
            if not math.isfinite(value) or value <= 0:
                raise GeckoLibGenerationContractError(
                    f"Existing GeckoLib entity record {field} must be positive and finite."
                )
        if str(record.get("archetype")) not in _ARCHETYPES:
            raise GeckoLibGenerationContractError(
                "Existing GeckoLib entity record archetype is invalid."
            )
        if str(record.get("behavior")) not in _BEHAVIORS:
            raise GeckoLibGenerationContractError(
                "Existing GeckoLib entity record behavior is invalid."
            )
        if str(record.get("spawn_group")) not in _SPAWN_GROUPS:
            raise GeckoLibGenerationContractError(
                "Existing GeckoLib entity record spawn_group is invalid."
            )


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
    "GeckoLibEntityGenerationInputs",
    "GeckoLibGenerationContractError",
    "GeckoLibGenerationTarget",
    "geckolib_entity_inputs_from_module_config",
    "preflight_geckolib_generation_target",
    "validate_existing_geckolib_records",
    "validate_geckolib_entity_inputs",
    "validate_geckolib_project_preflight",
]
