from __future__ import annotations

"""Strict semantic contracts for deterministic Minecraft content generation.

The generator is an executor, not a designer. Required semantic fields are owned by the
Minecraft template catalog; this module validates and lowers those fields without adding
new design decisions.
"""

import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache, wraps
from typing import Any

from .task_template_catalog import load_template

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_RESOURCE_ID = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")
_COMMAND_LITERAL = re.compile(r"^[a-z0-9_]+$")

SUPPORTED_GENERATED_KINDS = frozenset(
    {
        "item", "block", "tool", "weapon", "armor", "food", "crop", "machine",
        "effect", "enchantment", "command", "recipe", "advancement", "loot",
    }
)
_PRESENTATION_KINDS = frozenset(
    {"item", "block", "tool", "weapon", "armor", "food", "crop", "machine", "effect", "enchantment"}
)
_VISUAL_TEXTURE_KINDS = frozenset(
    {"item", "block", "tool", "weapon", "armor", "food", "crop", "machine"}
)


@lru_cache(maxsize=1)
def _generation_manifest() -> dict[str, Any]:
    manifest = load_template("minecraft/generation_contract")
    if manifest.get("execution") != "contract":
        raise ValueError("MINECRAFT_GENERATION_CONTRACT_INVALID: execution")
    kinds = manifest.get("kinds")
    if not isinstance(kinds, dict) or set(kinds) != SUPPORTED_GENERATED_KINDS:
        raise ValueError("MINECRAFT_GENERATION_CONTRACT_INVALID: kind coverage")
    for kind, spec in kinds.items():
        if not isinstance(spec, dict):
            raise ValueError(f"MINECRAFT_GENERATION_CONTRACT_INVALID: {kind}")
        fields = spec.get("required_fields")
        if (
            not isinstance(fields, list)
            or not fields
            or len(fields) != len(set(fields))
            or any(not isinstance(field, str) or not field for field in fields)
        ):
            raise ValueError(f"MINECRAFT_GENERATION_CONTRACT_INVALID: {kind}.required_fields")
    return manifest


def required_generation_fields(kind: str) -> tuple[str, ...]:
    spec = _generation_manifest()["kinds"].get(kind)
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported deterministic Minecraft module kind: {kind!r}")
    return tuple(spec["required_fields"])


def generation_authority(kind: str) -> str:
    spec = _generation_manifest()["kinds"].get(kind)
    if not isinstance(spec, dict):
        raise ValueError(f"Unsupported deterministic Minecraft module kind: {kind!r}")
    return str(spec.get("authority", "design"))


def _require_text(config: Mapping[str, Any], key: str, module_id: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_REQUIRED: {module_id}.{key}")
    return value.strip()


def _require_number(
    config: Mapping[str, Any], key: str, module_id: str, *, minimum: float | None = None
) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
    return numeric


def _require_int(
    config: Mapping[str, Any], key: str, module_id: str, *, minimum: int | None = None
) -> int:
    value = config.get(key)
    if type(value) is not int:
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
    if minimum is not None and value < minimum:
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
    return value


def _validate_color(config: Mapping[str, Any], key: str, module_id: str) -> str:
    value = _require_text(config, key, module_id)
    if not _HEX_COLOR.fullmatch(value):
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
    return value


def validate_generation_config(kind: str, module_id: str, config: Mapping[str, Any]) -> None:
    """Reject generation requests that would make the generator invent semantics."""
    if kind not in SUPPORTED_GENERATED_KINDS:
        raise ValueError(f"Unsupported deterministic Minecraft module kind: {kind!r}")
    if not isinstance(config, Mapping):
        raise ValueError(f"MINECRAFT_GENERATION_CONFIG_INVALID: {module_id}")
    missing = [field for field in required_generation_fields(kind) if field not in config]
    if missing:
        raise ValueError(f"MINECRAFT_GENERATION_FIELDS_REQUIRED: {module_id} missing {missing}")

    if kind in _PRESENTATION_KINDS:
        _require_text(config, "display_name", module_id)
    if kind in _VISUAL_TEXTURE_KINDS:
        _validate_color(config, "main_color", module_id)
    if "seed_color" in config:
        _validate_color(config, "seed_color", module_id)
    if kind == "effect":
        _validate_color(config, "color", module_id)

    if kind in {"tool", "weapon"}:
        _require_int(config, "attack_damage", module_id, minimum=0)
        _require_number(config, "attack_speed", module_id)
    elif kind == "food":
        _require_int(config, "hunger", module_id, minimum=0)
        _require_number(config, "saturation", module_id, minimum=0.0)
    elif kind == "armor":
        if _require_text(config, "slot", module_id).lower() not in {"helmet", "chestplate", "leggings", "boots"}:
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.slot")
    elif kind == "block":
        _require_number(config, "hardness", module_id, minimum=0.0)
    elif kind == "machine":
        for key in ("input_item", "output_item"):
            if not _RESOURCE_ID.fullmatch(_require_text(config, key, module_id)):
                raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
        _require_int(config, "output_count", module_id, minimum=1)
        _require_int(config, "processing_ticks", module_id, minimum=1)
    elif kind == "enchantment":
        _require_int(config, "max_level", module_id, minimum=1)
    elif kind == "command":
        if not _COMMAND_LITERAL.fullmatch(_require_text(config, "literal", module_id)):
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.literal")
        _require_text(config, "message", module_id)
        permission = _require_int(config, "permission_level", module_id, minimum=0)
        if permission > 4:
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.permission_level")
    elif generation_authority(kind) == "artifact_resource":
        payload = config.get("json")
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.json")


def lower_generation_config(kind: str, module_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Lower canonical design fields to legacy generator names without new reasoning."""
    validate_generation_config(kind, module_id, config)
    lowered = dict(config)
    display_name = lowered.get("display_name")
    if kind in _PRESENTATION_KINDS and isinstance(display_name, str):
        lowered.setdefault("display_name_en", display_name)
        lowered.setdefault("display_name_ko", display_name)
    if kind in _VISUAL_TEXTURE_KINDS:
        canonical_color = str(lowered["main_color"])
        legacy_color = lowered.get("color")
        if legacy_color is not None and str(legacy_color) != canonical_color:
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_CONFLICT: {module_id}.main_color/color")
        lowered["color"] = canonical_color
    return lowered


def config_schema_for_kind(kind: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "display_name": {"type": "string", "minLength": 1, "maxLength": 160},
        "display_name_ko": {"type": "string", "minLength": 1, "maxLength": 160},
        "main_color": {"type": "string", "pattern": _HEX_COLOR.pattern},
        "color": {"type": "string", "pattern": _HEX_COLOR.pattern},
        "seed_color": {"type": "string", "pattern": _HEX_COLOR.pattern},
        "hardness": {"type": "number", "minimum": 0},
        "attack_damage": {"type": "integer", "minimum": 0},
        "attack_speed": {"type": "number"},
        "hunger": {"type": "integer", "minimum": 0},
        "saturation": {"type": "number", "minimum": 0},
        "slot": {"type": "string", "enum": ["helmet", "chestplate", "leggings", "boots"]},
        "input_item": {"type": "string", "pattern": _RESOURCE_ID.pattern},
        "output_item": {"type": "string", "pattern": _RESOURCE_ID.pattern},
        "output_count": {"type": "integer", "minimum": 1},
        "processing_ticks": {"type": "integer", "minimum": 1},
        "max_level": {"type": "integer", "minimum": 1},
        "literal": {"type": "string", "pattern": _COMMAND_LITERAL.pattern},
        "message": {"type": "string", "minLength": 1, "maxLength": 512},
        "permission_level": {"type": "integer", "minimum": 0, "maximum": 4},
        "json": {"type": "object", "minProperties": 1},
    }
    return {
        "type": "object",
        "required": list(required_generation_fields(kind)),
        "additionalProperties": True,
        "properties": properties,
    }


def install_generation_guard(extended_module: Any) -> None:
    """Guard every normal extended-generator call, not only the model tool path."""
    original = extended_module.generate_extended_content
    if getattr(original, "_mmm_strict_generation_fields", False):
        return

    @wraps(original)
    def guarded_generate_extended_content(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if args:
            return original(*args, **kwargs)
        raw_modules = kwargs.get("modules")
        if raw_modules is None:
            return original(**kwargs)
        modules = tuple(raw_modules)
        lowered = []
        supported = frozenset(str(value) for value in extended_module._SUPPORTED)
        for module in modules:
            if module.kind not in supported:
                lowered.append(module)
                continue
            config = lower_generation_config(module.kind, module.module_id, module.config)
            lowered.append(replace(module, config=config))
        kwargs["modules"] = tuple(lowered)
        return original(**kwargs)

    guarded_generate_extended_content._mmm_strict_generation_fields = True  # type: ignore[attr-defined]
    guarded_generate_extended_content.__wrapped__ = original  # type: ignore[attr-defined]
    extended_module.generate_extended_content = guarded_generate_extended_content


def _install_loaded_generator_guard() -> None:
    loaded = sys.modules.get("minecraft_mod_ai.extended_content_generator")
    if loaded is not None:
        install_generation_guard(loaded)


_install_loaded_generator_guard()


__all__ = [
    "SUPPORTED_GENERATED_KINDS",
    "config_schema_for_kind",
    "generation_authority",
    "install_generation_guard",
    "lower_generation_config",
    "required_generation_fields",
    "validate_generation_config",
]
