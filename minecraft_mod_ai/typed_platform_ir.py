from __future__ import annotations

"""Typed platform-module contract for deterministic production backends."""

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .system_pack_validation import validate_system_modules

PLATFORM_CONTENT_KINDS = frozenset({
    "item", "block", "tool", "weapon", "armor", "food", "crop", "machine",
    "effect", "enchantment", "command", "recipe", "advancement", "loot",
})
PLATFORM_SYSTEM_KIND_TO_PACK = {
    "quest": "quest-system",
    "class": "class-skill-system",
    "skill": "class-skill-system",
    "economy": "economy-shop",
    "shop": "economy-shop",
    "gui": "gui-networking",
    "networking": "gui-networking",
    "party": "party-guild",
    "guild": "party-guild",
}
PLATFORM_KINDS = frozenset(
    set(PLATFORM_CONTENT_KINDS) | set(PLATFORM_SYSTEM_KIND_TO_PACK)
)

_ID = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_RESOURCE = re.compile(r"^[a-z0-9_.-]+:[a-z0-9_./-]+$")
_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


def _schema(properties: Mapping[str, Any], *, required: Sequence[str] = ()) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": False,
    }


_STRING = {"type": "string", "minLength": 1, "maxLength": 256}
_RESOURCE_ID = {
    "type": "string",
    "pattern": r"^[a-z0-9_.-]+:[a-z0-9_./-]+$",
}
_DISPLAY = {"type": "string", "minLength": 1, "maxLength": 128}
_POSITIVE_INT = {"type": "integer", "minimum": 1}
_NONNEG_INT = {"type": "integer", "minimum": 0}
_FINITE_NUMBER = {"type": "number"}


def platform_config_schema(kind: str) -> dict[str, Any]:
    common = {
        "display_name_en": _DISPLAY,
        "display_name_ko": _DISPLAY,
        "ingredients": {
            "type": "array",
            "maxItems": 64,
            "items": _RESOURCE_ID,
        },
    }
    if kind == "item":
        return _schema(common)
    if kind == "block":
        return _schema({**common, "hardness": _FINITE_NUMBER})
    if kind == "food":
        return _schema({
            **common,
            "hunger": _NONNEG_INT,
            "saturation": _FINITE_NUMBER,
        })
    if kind in {"weapon", "tool"}:
        return _schema({
            **common,
            "attack_damage": {"type": "integer"},
            "attack_speed": _FINITE_NUMBER,
        })
    if kind == "armor":
        return _schema({
            **common,
            "slot": {
                "type": "string",
                "enum": ["helmet", "chestplate", "leggings", "boots"],
            },
        })
    if kind == "machine":
        return _schema({
            **common,
            "input_item": _RESOURCE_ID,
            "output_item": _RESOURCE_ID,
            "output_count": _POSITIVE_INT,
            "processing_ticks": _POSITIVE_INT,
        })
    if kind == "crop":
        return _schema(common)
    if kind == "effect":
        return _schema({
            **common,
            "color": {
                "type": "string",
                "pattern": r"^#[0-9A-Fa-f]{6}$",
            },
        })
    if kind == "enchantment":
        return _schema({**common, "max_level": _POSITIVE_INT})
    if kind == "command":
        return _schema({
            "literal": {
                "type": "string",
                "pattern": r"^[a-z0-9_]+$",
                "minLength": 1,
                "maxLength": 64,
            },
            "message": {"type": "string", "maxLength": 512},
            "permission_level": {
                "type": "integer",
                "minimum": 0,
                "maximum": 4,
            },
        })
    if kind in {"recipe", "advancement", "loot"}:
        return _schema({
            "json": {
                "type": "object",
                "maxProperties": 128,
            },
        }, required=("json",))
    if kind == "quest":
        return _schema({
            "objective": {"type": "string", "enum": ["kill", "break", "manual"]},
            "target": _STRING,
            "required": _POSITIVE_INT,
            "reward_item": _RESOURCE_ID,
            "reward_count": _POSITIVE_INT,
            "reward_currency": {"type": "number", "minimum": 0},
        })
    if kind == "class":
        return _schema({"display_name": _DISPLAY})
    if kind == "skill":
        return _schema({
            "required_class": {
                "type": "string",
                "pattern": r"^[a-z][a-z0-9_]{1,63}$",
            },
            "effect": _RESOURCE_ID,
            "duration_ticks": _POSITIVE_INT,
            "amplifier": {"type": "integer", "minimum": 0, "maximum": 255},
            "cooldown_ticks": _POSITIVE_INT,
        })
    if kind == "economy":
        return _schema({"initial_balance": {"type": "number", "minimum": 0}})
    if kind == "shop":
        return _schema({
            "entries": {
                "type": "array",
                "minItems": 1,
                "maxItems": 128,
                "items": _schema({
                    "id": {
                        "type": "string",
                        "pattern": r"^[a-z][a-z0-9_]{1,63}$",
                    },
                    "item": _RESOURCE_ID,
                    "count": _POSITIVE_INT,
                    "price": {"type": "number", "minimum": 0},
                }, required=("id", "item", "price")),
            },
        }, required=("entries",))
    if kind == "gui":
        return _schema({
            "template": {"type": "string", "enum": ["read_only_menu"]},
            "title": _DISPLAY,
            "rows": {"type": "integer", "minimum": 1, "maximum": 6},
            "entries": {
                "type": "array",
                "maxItems": 54,
                "items": _schema({
                    "slot": {"type": "integer", "minimum": 0, "maximum": 53},
                    "item": _RESOURCE_ID,
                    "count": _POSITIVE_INT,
                }, required=("slot", "item")),
            },
        }, required=("template",))
    if kind == "networking":
        action = {
            "oneOf": [
                _schema({
                    "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]{1,63}$"},
                    "type": {"const": "message"},
                    "message": _STRING,
                }, required=("id", "type", "message")),
                _schema({
                    "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]{1,63}$"},
                    "type": {"const": "grant_item"},
                    "item": _RESOURCE_ID,
                    "count": _POSITIVE_INT,
                }, required=("id", "type", "item")),
                _schema({
                    "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]{1,63}$"},
                    "type": {"const": "status_effect"},
                    "effect": _RESOURCE_ID,
                    "duration_ticks": _POSITIVE_INT,
                    "amplifier": {"type": "integer", "minimum": 0, "maximum": 255},
                }, required=("id", "type", "effect")),
            ],
        }
        return _schema({
            "template": {"type": "string", "enum": ["validated_action_channel"]},
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": 128,
                "items": action,
            },
        }, required=("template", "actions"))
    if kind in {"party", "guild"}:
        return _schema({"display_name": _DISPLAY})
    raise ValueError(f"TYPED_PLATFORM_KIND_UNSUPPORTED: {kind!r}")


def _json_scalar_tree(value: Any, where: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{where}: non-finite number")
        return
    if isinstance(value, list):
        if len(value) > 256:
            raise ValueError(f"{where}: array too large")
        for index, item in enumerate(value):
            _json_scalar_tree(item, f"{where}[{index}]")
        return
    if isinstance(value, Mapping):
        if len(value) > 256:
            raise ValueError(f"{where}: object too large")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{where}: object key must be string")
            _json_scalar_tree(item, f"{where}.{key}")
        return
    raise ValueError(f"{where}: unsupported JSON value {type(value).__name__}")


def _validate_content_config(kind: str, config: Mapping[str, Any], module_id: str) -> None:
    common = {"display_name_en", "display_name_ko", "ingredients"}
    allowed_by_kind = {
        "item": common,
        "block": common | {"hardness"},
        "food": common | {"hunger", "saturation"},
        "weapon": common | {"attack_damage", "attack_speed"},
        "tool": common | {"attack_damage", "attack_speed"},
        "armor": common | {"slot"},
        "machine": common | {"input_item", "output_item", "output_count", "processing_ticks"},
        "crop": common,
        "effect": common | {"color"},
        "enchantment": common | {"max_level"},
        "command": {"literal", "message", "permission_level"},
        "recipe": {"json"},
        "advancement": {"json"},
        "loot": {"json"},
    }
    unknown = set(config) - allowed_by_kind[kind]
    if unknown:
        raise ValueError(
            f"TYPED_PLATFORM_CONFIG_INVALID: {module_id} has unsupported fields {sorted(unknown)}"
        )

    for field in ("display_name_en", "display_name_ko", "message"):
        if field in config and not isinstance(config[field], str):
            raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.{field}")
    if "ingredients" in config:
        items = config["ingredients"]
        if not isinstance(items, list) or len(items) > 64 or any(
            not isinstance(item, str) or not _RESOURCE.fullmatch(item) for item in items
        ):
            raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.ingredients")
    for field in ("hardness", "saturation", "attack_speed"):
        if field in config and (
            isinstance(config[field], bool)
            or not isinstance(config[field], (int, float))
            or not math.isfinite(float(config[field]))
        ):
            raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.{field}")
    for field in ("hunger", "attack_damage", "output_count", "processing_ticks", "max_level", "permission_level"):
        if field in config and type(config[field]) is not int:
            raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.{field}")
    if kind == "armor" and config.get("slot", "chestplate") not in {
        "helmet", "chestplate", "leggings", "boots"
    }:
        raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.slot")
    if kind == "machine":
        for field in ("input_item", "output_item"):
            value = str(config.get(field, "minecraft:iron_ingot"))
            if not _RESOURCE.fullmatch(value):
                raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.{field}")
    if kind == "effect" and "color" in config and not _HEX.fullmatch(str(config["color"])):
        raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.color")
    if kind == "command" and "literal" in config:
        if not re.fullmatch(r"[a-z0-9_]+", str(config["literal"])):
            raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.literal")
    if kind in {"recipe", "advancement", "loot"}:
        payload = config.get("json")
        if not isinstance(payload, Mapping):
            raise ValueError(f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.json")
        _json_scalar_tree(payload, f"{module_id}.json")


def validate_platform_modules(raw_modules: Any) -> list[dict[str, Any]]:
    if raw_modules is None:
        return []
    if not isinstance(raw_modules, Sequence) or isinstance(
        raw_modules, (str, bytes, bytearray)
    ):
        raise ValueError("typed_plan_ir.platform_modules: expected array")

    modules: list[dict[str, Any]] = []
    seen: set[str] = set()
    system_groups: dict[str, list[dict[str, Any]]] = {}
    for index, raw in enumerate(raw_modules):
        where = f"typed_plan_ir.platform_modules[{index}]"
        if not isinstance(raw, Mapping):
            raise ValueError(f"{where}: expected object")
        if set(raw) != {"module_id", "kind", "config", "covers"}:
            raise ValueError(f"{where}: invalid fields")
        module_id = str(raw["module_id"])
        kind = str(raw["kind"])
        if not _ID.fullmatch(module_id) or module_id in seen:
            raise ValueError(f"{where}: invalid or duplicate module_id {module_id!r}")
        if kind not in PLATFORM_KINDS:
            raise ValueError(f"{where}: unsupported kind {kind!r}")
        config = raw["config"]
        if not isinstance(config, Mapping):
            raise ValueError(f"{where}.config: expected object")
        covers = raw["covers"]
        if not isinstance(covers, Sequence) or isinstance(
            covers, (str, bytes, bytearray)
        ) or not covers:
            raise ValueError(f"{where}.covers: expected non-empty array")
        normalized_covers = []
        for cover in covers:
            if not isinstance(cover, str) or not cover.strip():
                raise ValueError(f"{where}.covers: invalid coverage ref")
            normalized_covers.append(cover.strip())

        seen.add(module_id)
        record = {
            "module_id": module_id,
            "kind": kind,
            "config": deepcopy(dict(config)),
            "covers": list(dict.fromkeys(normalized_covers)),
        }
        if kind in PLATFORM_CONTENT_KINDS:
            _validate_content_config(kind, config, module_id)
        else:
            pack = PLATFORM_SYSTEM_KIND_TO_PACK[kind]
            system_groups.setdefault(pack, []).append({
                "module_id": module_id,
                "kind": kind,
                "config": deepcopy(dict(config)),
                "depends_on": [],
                "required_gates": [],
            })
        modules.append(record)

    for pack_id, group in system_groups.items():
        validate_system_modules(pack_id, group)
    return modules


def platform_module_authoring_schema() -> dict[str, Any]:
    """Schema for one host-supported platform module."""

    branches = []
    for kind in sorted(PLATFORM_KINDS):
        branches.append(_schema({
            "module_id": {
                "type": "string",
                "pattern": r"^[a-z][a-z0-9_]{1,63}$",
            },
            "kind": {"const": kind},
            "config": platform_config_schema(kind),
            "covers": {
                "type": "array",
                "minItems": 1,
                "maxItems": 64,
                "items": {"type": "string", "minLength": 1, "maxLength": 256},
            },
        }, required=("module_id", "kind", "config", "covers")))
    return {"oneOf": branches}


__all__ = [
    "PLATFORM_CONTENT_KINDS",
    "PLATFORM_KINDS",
    "PLATFORM_SYSTEM_KIND_TO_PACK",
    "platform_config_schema",
    "platform_module_authoring_schema",
    "validate_platform_modules",
]
