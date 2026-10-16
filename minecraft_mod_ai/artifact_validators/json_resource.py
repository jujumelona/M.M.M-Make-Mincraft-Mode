from __future__ import annotations

"""JSON resource validation for models, recipes, tags, loot tables, and lang files."""

import json
from typing import Any


class JsonResourceValidationError(ValueError):
    pass


def validate_json_resource(content: str, *, resource_kind: str = "") -> dict[str, Any]:
    """Validate that content is well-formed JSON conforming to Minecraft resource norms."""
    if not isinstance(content, str) or not content.strip():
        raise JsonResourceValidationError("JSON_EMPTY: Resource content cannot be empty")

    try:
        data = json.loads(content)
    except Exception as exc:
        raise JsonResourceValidationError(f"JSON_SYNTAX: Invalid JSON: {exc}") from exc

    if not isinstance(data, (dict, list)):
        raise JsonResourceValidationError(
            "JSON_ROOT: Minecraft resources must be JSON objects or arrays"
        )

    return {
        "status": "PASS",
        "resource_kind": resource_kind,
        "key_count": len(data) if isinstance(data, dict) else len(data),
    }

 
def validate_dimension_definition(content: str, *, base_dir=None) -> dict[str, Any]:
    """Check a 26.x dimension instance and locally resolvable registry links.

    This is not worldgen runtime certification. A custom type/setting must
    exist in the project before an artifact can claim file-level validity.
    """
    import re
    from pathlib import Path

    validate_json_resource(content, resource_kind="dimension")
    value = json.loads(content)
    if not isinstance(value, dict) or set(value) != {"type", "generator"}:
        raise JsonResourceValidationError("DIMENSION_ROOT: type and generator required")
    id_pattern = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")
    deferred_refs = []

    def ref(value, kind, vanilla=()):
        if not isinstance(value, str) or not id_pattern.fullmatch(value):
            raise JsonResourceValidationError(f"DIMENSION_REFERENCE_INVALID: {kind}")
        if value in vanilla:
            return
        namespace, name = value.split(":", 1)
        if base_dir is None or namespace == "minecraft":
            # Vanilla IDs require target-game registry validation; do not
            # mistake absence from the mod's own data pack for a missing file.
            deferred_refs.append(value)
            return
        root = Path(base_dir).resolve()
        target = (root / "src/main/resources/data" / namespace / kind /
                  (name + ".json")).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise JsonResourceValidationError(f"DIMENSION_REFERENCE_UNRESOLVED: {kind}:{value}")
        validate_json_resource(target.read_text(encoding="utf-8"), resource_kind=kind)

    ref(value["type"], "dimension_type", (
        "minecraft:overworld", "minecraft:overworld_caves",
        "minecraft:the_nether", "minecraft:the_end",
    ))
    generator = value["generator"]
    if not isinstance(generator, dict):
        raise JsonResourceValidationError("DIMENSION_GENERATOR_REQUIRED")
    kind = generator.get("type")
    if kind not in {"minecraft:noise", "minecraft:flat", "minecraft:debug"}:
        raise JsonResourceValidationError("DIMENSION_GENERATOR_TYPE_INVALID")
    if kind == "minecraft:noise":
        settings = generator.get("settings")
        if isinstance(settings, str):
            ref(settings, "worldgen/noise_settings", (
                "minecraft:overworld", "minecraft:large_biomes",
                "minecraft:amplified", "minecraft:nether", "minecraft:end",
                "minecraft:caves", "minecraft:floating_islands",
            ))
        elif not isinstance(settings, dict) or not settings:
            raise JsonResourceValidationError("DIMENSION_NOISE_SETTINGS_REQUIRED")
        biome_source = generator.get("biome_source")
        if not isinstance(biome_source, dict) or not isinstance(biome_source.get("type"), str):
            raise JsonResourceValidationError("DIMENSION_BIOME_SOURCE_REQUIRED")
        source_type = biome_source["type"]
        if source_type == "minecraft:fixed":
            ref(biome_source.get("biome"), "worldgen/biome")
        elif source_type == "minecraft:multi_noise":
            if "preset" in biome_source:
                ref(biome_source["preset"], "worldgen/multi_noise_biome_source_parameter_list",
                    ("minecraft:overworld", "minecraft:nether"))
            elif not isinstance(biome_source.get("biomes"), list) or not biome_source["biomes"]:
                raise JsonResourceValidationError("DIMENSION_MULTI_NOISE_BIOMES_REQUIRED")
        elif source_type == "minecraft:checkerboard":
            biomes = biome_source.get("biomes")
            if not isinstance(biomes, list) or not biomes:
                raise JsonResourceValidationError("DIMENSION_CHECKERBOARD_BIOMES_REQUIRED")
            for biome in biomes:
                ref(biome, "worldgen/biome")
        elif source_type != "minecraft:the_end":
            raise JsonResourceValidationError("DIMENSION_BIOME_SOURCE_TYPE_INVALID")
    elif kind == "minecraft:flat":
        settings = generator.get("settings")
        if (not isinstance(settings, dict) or not isinstance(settings.get("layers"), list)
                or not isinstance(settings.get("biome"), str)):
            raise JsonResourceValidationError("DIMENSION_FLAT_SETTINGS_REQUIRED")
        ref(settings["biome"], "worldgen/biome")
    return {
        "validator": "dimension_definition",
        "status": "PASS",
        "reference_validation": "DEFERRED" if deferred_refs else "PROJECT_CHECKED",
        "unverified_registry_ids": sorted(set(deferred_refs)),
        "generator_type": kind,
    }
