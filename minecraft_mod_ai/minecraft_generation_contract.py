from __future__ import annotations

"""Strict semantic contracts for deterministic Minecraft content generation.

The generator is an executor, not a designer.  Every gameplay- or presentation-relevant
value that changes generated output must be supplied explicitly before generation.
"""

import re
from collections.abc import Mapping
from typing import Any

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")
_RESOURCE_ID = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")
_COMMAND_LITERAL = re.compile(r"^[a-z0-9_]+$")

SUPPORTED_GENERATED_KINDS = frozenset(
    {
        "item",
        "block",
        "tool",
        "weapon",
        "armor",
        "food",
        "crop",
        "machine",
        "effect",
        "enchantment",
        "command",
        "recipe",
        "advancement",
        "loot",
    }
)

# Fields whose absence previously caused the generator to choose semantic defaults.
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "item": ("display_name_en", "display_name_ko", "color"),
    "block": ("display_name_en", "display_name_ko", "color", "hardness"),
    "tool": (
        "display_name_en",
        "display_name_ko",
        "color",
        "attack_damage",
        "attack_speed",
    ),
    "weapon": (
        "display_name_en",
        "display_name_ko",
        "color",
        "attack_damage",
        "attack_speed",
    ),
    "armor": ("display_name_en", "display_name_ko", "color", "slot"),
    "food": (
        "display_name_en",
        "display_name_ko",
        "color",
        "hunger",
        "saturation",
    ),
    "crop": (
        "display_name_en",
        "display_name_ko",
        "color",
        "seed_color",
    ),
    "machine": (
        "display_name_en",
        "display_name_ko",
        "color",
        "input_item",
        "output_item",
        "output_count",
        "processing_ticks",
    ),
    "effect": ("display_name_en", "display_name_ko", "color"),
    "enchantment": ("display_name_en", "display_name_ko", "max_level"),
    "command": ("literal", "message", "permission_level"),
    # These are already payload-driven by _data_only_resource; no presentation fallback is used.
    "recipe": (),
    "advancement": (),
    "loot": (),
}


def required_generation_fields(kind: str) -> tuple[str, ...]:
    try:
        return _REQUIRED_FIELDS[kind]
    except KeyError as exc:
        raise ValueError(f"Unsupported deterministic Minecraft module kind: {kind!r}") from exc


def _require_text(config: Mapping[str, Any], key: str, module_id: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_REQUIRED: {module_id}.{key}")
    return value.strip()


def _require_number(
    config: Mapping[str, Any],
    key: str,
    module_id: str,
    *,
    minimum: float | None = None,
) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
    numeric = float(value)
    if minimum is not None and numeric < minimum:
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
    return numeric


def _require_int(
    config: Mapping[str, Any],
    key: str,
    module_id: str,
    *,
    minimum: int | None = None,
) -> int:
    value = config.get(key)
    if type(value) is not int:
        raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
    if minimum is not None and value < minimum:
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
        raise ValueError(
            f"MINECRAFT_GENERATION_FIELDS_REQUIRED: {module_id} missing {missing}"
        )

    for key in ("display_name_en", "display_name_ko"):
        if key in config:
            _require_text(config, key, module_id)
    for key in ("color", "seed_color"):
        if key in config:
            value = _require_text(config, key, module_id)
            if not _HEX_COLOR.fullmatch(value):
                raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")

    if kind in {"tool", "weapon"}:
        _require_int(config, "attack_damage", module_id, minimum=0)
        _require_number(config, "attack_speed", module_id)
    elif kind == "food":
        _require_int(config, "hunger", module_id, minimum=0)
        _require_number(config, "saturation", module_id, minimum=0.0)
    elif kind == "armor":
        slot = _require_text(config, "slot", module_id).lower()
        if slot not in {"helmet", "chestplate", "leggings", "boots"}:
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.slot")
    elif kind == "block":
        _require_number(config, "hardness", module_id, minimum=0.0)
    elif kind == "machine":
        for key in ("input_item", "output_item"):
            value = _require_text(config, key, module_id)
            if not _RESOURCE_ID.fullmatch(value):
                raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.{key}")
        _require_int(config, "output_count", module_id, minimum=1)
        _require_int(config, "processing_ticks", module_id, minimum=1)
    elif kind == "enchantment":
        _require_int(config, "max_level", module_id, minimum=1)
    elif kind == "command":
        literal = _require_text(config, "literal", module_id)
        if not _COMMAND_LITERAL.fullmatch(literal):
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.literal")
        _require_text(config, "message", module_id)
        _require_int(config, "permission_level", module_id, minimum=0)


def config_schema_for_kind(kind: str) -> dict[str, Any]:
    """Return the bounded model-facing schema for one generated module kind."""
    required = list(required_generation_fields(kind))
    properties: dict[str, Any] = {
        "display_name_en": {"type": "string", "minLength": 1, "maxLength": 160},
        "display_name_ko": {"type": "string", "minLength": 1, "maxLength": 160},
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
        "permission_level": {"type": "integer", "minimum": 0},
    }
    return {
        "type": "object",
        "required": required,
        # Preserve specialized payload keys used by recipe/advancement/loot and future
        # reviewed generators; strict semantic requiredness is enforced above.
        "additionalProperties": True,
        "properties": properties,
    }


__all__ = [
    "SUPPORTED_GENERATED_KINDS",
    "config_schema_for_kind",
    "required_generation_fields",
    "validate_generation_config",
]
