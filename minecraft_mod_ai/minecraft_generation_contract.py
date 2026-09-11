from __future__ import annotations

"""Strict semantic contracts for deterministic Minecraft content generation.

The generator is an executor, not a designer. Every gameplay- or presentation-relevant
value that changes generated output must be supplied explicitly before generation.
Canonical design-facing names are lowered to legacy generator field names here so a
small model never has to reproduce generator implementation details.
"""

import re
import sys
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
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

_PRESENTATION_KINDS = frozenset(
    {"item", "block", "tool", "weapon", "armor", "food", "crop", "machine", "effect", "enchantment"}
)
_VISUAL_TEXTURE_KINDS = frozenset(
    {"item", "block", "tool", "weapon", "armor", "food", "crop", "machine"}
)

# Canonical generation inputs. Legacy generator-only names (display_name_en/ko and
# color for textured content) are derived by lower_generation_config and are not
# separate reasoning obligations for the model.
_REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    "item": ("display_name", "main_color"),
    "block": ("display_name", "main_color", "hardness"),
    "tool": ("display_name", "main_color", "attack_damage", "attack_speed"),
    "weapon": ("display_name", "main_color", "attack_damage", "attack_speed"),
    "armor": ("display_name", "main_color", "slot"),
    "food": ("display_name", "main_color", "hunger", "saturation"),
    "crop": ("display_name", "main_color", "seed_color"),
    "machine": (
        "display_name",
        "main_color",
        "input_item",
        "output_item",
        "output_count",
        "processing_ticks",
    ),
    "effect": ("display_name", "color"),
    "enchantment": ("display_name", "max_level"),
    "command": ("literal", "message", "permission_level"),
    # Data-only artifacts must carry an explicit serialized payload. Using the whole
    # config as a fallback leaks orchestration metadata into Minecraft JSON.
    "recipe": ("json",),
    "advancement": ("json",),
    "loot": ("json",),
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
        raise ValueError(
            f"MINECRAFT_GENERATION_FIELDS_REQUIRED: {module_id} missing {missing}"
        )

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
        permission = _require_int(config, "permission_level", module_id, minimum=0)
        if permission > 4:
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.permission_level")
    elif kind in {"recipe", "advancement", "loot"}:
        payload = config.get("json")
        if not isinstance(payload, dict) or not payload:
            raise ValueError(f"MINECRAFT_GENERATION_FIELD_INVALID: {module_id}.json")


def lower_generation_config(kind: str, module_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Lower canonical design fields to the existing generator without new reasoning."""
    validate_generation_config(kind, module_id, config)
    lowered = dict(config)
    display_name = lowered.get("display_name")
    if kind in _PRESENTATION_KINDS and isinstance(display_name, str):
        # English is the authored canonical name today. Korean localization remains a
        # separate optional authority; absent translation means deterministic identity,
        # never a generated translation guess.
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
    """Return the bounded small-model-facing schema for one generated module kind."""
    required = list(required_generation_fields(kind))
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
        "required": required,
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
            # The wrapped generator is keyword-only. Preserve that API rather than
            # silently accepting a second calling convention.
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
    """Compose with the generator already imported by runtime_bootstrap."""
    loaded = sys.modules.get("minecraft_mod_ai.extended_content_generator")
    if loaded is not None:
        install_generation_guard(loaded)


_install_loaded_generator_guard()


__all__ = [
    "SUPPORTED_GENERATED_KINDS",
    "config_schema_for_kind",
    "install_generation_guard",
    "lower_generation_config",
    "required_generation_fields",
    "validate_generation_config",
]
