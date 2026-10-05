from __future__ import annotations

"""Typed platform-module contract for deterministic production backends."""

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from .authored_content_contract import RESOURCE_POLICY_CONCERNS
from .platform_backend_contract import (
    ENTITY_PIPELINE_KINDS as PLATFORM_ENTITY_KINDS,
    EXTENDED_CONTENT_KINDS as PLATFORM_CONTENT_KINDS,
    SYSTEM_KIND_TO_PACK as PLATFORM_SYSTEM_KIND_TO_PACK,
)
from .system_pack_validation import validate_system_modules

PLATFORM_HOST_SECTION_OWNER = {
    "authority_and_network": "network_sync",
    "persistence": "state_store",
}
PLATFORM_HOST_MODULE_IDS = {
    "state_store": "typed_state_store",
    "network_sync": "typed_network_sync",
    "resource_policy": "typed_resource_policy",
}
PLATFORM_HOST_KINDS = frozenset(PLATFORM_HOST_MODULE_IDS)
NETWORK_SYNC_STATEFUL_REFS = frozenset({
    "authority_and_network.payloads",
    "authority_and_network.synchronization",
    "authority_and_network.reconnection",
})


def host_platform_kind_for_ref(ref: str) -> str | None:
    """Return the single host-owned backend for a canonical concern ref."""

    section, dot, concern = str(ref or "").partition(".")
    if not dot:
        return None
    owner = PLATFORM_HOST_SECTION_OWNER.get(section)
    if owner is not None:
        return owner
    if section == "resources_and_ui" and concern in RESOURCE_POLICY_CONCERNS:
        return "resource_policy"
    return None


def network_sync_requires_state(covers: Sequence[str]) -> bool:
    """Whether a network policy module must bind canonical state transport."""

    return bool(
        NETWORK_SYNC_STATEFUL_REFS
        & {str(ref).strip() for ref in covers if str(ref).strip()}
    )

PLATFORM_KINDS = frozenset(
    set(PLATFORM_CONTENT_KINDS)
    | set(PLATFORM_SYSTEM_KIND_TO_PACK)
    | set(PLATFORM_HOST_KINDS)
    | set(PLATFORM_ENTITY_KINDS)
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
        return _schema(
            common,
            required=("display_name_en", "display_name_ko"),
        )
    if kind == "block":
        return _schema(
            {**common, "hardness": _FINITE_NUMBER},
            required=("display_name_en", "display_name_ko", "hardness"),
        )
    if kind == "food":
        return _schema({
            **common,
            "hunger": _NONNEG_INT,
            "saturation": _FINITE_NUMBER,
        }, required=(
            "display_name_en",
            "display_name_ko",
            "hunger",
            "saturation",
        ))
    if kind in {"weapon", "tool"}:
        return _schema({
            **common,
            "attack_damage": {"type": "integer"},
            "attack_speed": _FINITE_NUMBER,
        }, required=(
            "display_name_en",
            "display_name_ko",
            "attack_damage",
            "attack_speed",
        ))
    if kind == "armor":
        return _schema({
            **common,
            "slot": {
                "type": "string",
                "enum": ["helmet", "chestplate", "leggings", "boots"],
            },
        }, required=("display_name_en", "display_name_ko", "slot"))
    if kind == "machine":
        return _schema({
            **common,
            "input_item": _RESOURCE_ID,
            "output_item": _RESOURCE_ID,
            "output_count": _POSITIVE_INT,
            "processing_ticks": _POSITIVE_INT,
        }, required=(
            "display_name_en",
            "display_name_ko",
            "input_item",
            "output_item",
            "output_count",
            "processing_ticks",
        ))
    if kind == "crop":
        return _schema(
            common,
            required=("display_name_en", "display_name_ko"),
        )
    if kind == "effect":
        return _schema({
            **common,
            "color": {
                "type": "string",
                "pattern": r"^#[0-9A-Fa-f]{6}$",
            },
        }, required=("display_name_en", "display_name_ko", "color"))
    if kind == "enchantment":
        return _schema(
            {**common, "max_level": _POSITIVE_INT},
            required=("display_name_en", "display_name_ko", "max_level"),
        )
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
        }, required=("literal", "message", "permission_level"))
    if kind in {"recipe", "advancement", "loot"}:
        return _schema({
            "json": {
                "type": "object",
                "maxProperties": 128,
            },
        }, required=("json",))
    if kind == "tag":
        return _schema({
            "registry": {
                "type": "string",
                "enum": ["items", "blocks", "entity_types", "fluids", "functions"],
            },
            "values": {
                "type": "array",
                "minItems": 1,
                "maxItems": 256,
                "items": _RESOURCE_ID,
            },
            "replace": {"type": "boolean"},
        }, required=("registry", "values"))
    if kind in PLATFORM_ENTITY_KINDS:
        return _schema({
            "max_health": {"type": "number", "exclusiveMinimum": 0},
            "attack_damage": {"type": "number", "minimum": 0},
            "movement_speed": {"type": "number", "exclusiveMinimum": 0},
            "follow_range": {"type": "number", "exclusiveMinimum": 0},
            "archetype": {
                "type": "string",
                "enum": [
                    "biped",
                    "quadruped",
                    "flying",
                    "serpentine",
                    "construct",
                ],
            },
            "behavior": {
                "type": "string",
                "enum": [
                    "hostile_melee",
                    "neutral_melee",
                    "passive",
                    "npc",
                ],
            },
            "entity_width": {"type": "number", "exclusiveMinimum": 0},
            "entity_height": {"type": "number", "exclusiveMinimum": 0},
            "spawn_group": {
                "type": "string",
                "enum": [
                    "monster",
                    "creature",
                    "ambient",
                    "water_creature",
                    "misc",
                ],
            },
            "main_color": {
                "type": "string",
                "pattern": r"^#[0-9A-Fa-f]{6}$",
            },
            "texture_width": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4096,
            },
            "texture_height": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4096,
            },
        }, required=(
            "max_health",
            "attack_damage",
            "movement_speed",
            "follow_range",
            "archetype",
            "behavior",
            "entity_width",
            "entity_height",
            "spawn_group",
            "main_color",
        ))
    if kind == "state_store":
        migration = _schema({
            "from_version": _STRING,
            "to_version": _STRING,
            "operation": {
                "type": "string",
                "enum": [
                    "preserve",
                    "rename_key",
                    "delete_key",
                    "set_default",
                ],
            },
            "source_key": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
            },
            "destination_key": {
                "type": "string",
                "minLength": 1,
                "maxLength": 128,
            },
            "value": {
                "type": ["string", "number", "integer", "boolean", "null"],
            },
        }, required=("from_version", "to_version", "operation"))
        return _schema({
            "namespace": {
                "type": "string",
                "pattern": r"^[a-z][a-z0-9_.-]{1,63}$",
            },
            "schema_version": _STRING,
            "migrations": {
                "type": "array",
                "maxItems": 64,
                "items": migration,
            },
            "malformed_policy": {
                "type": "string",
                "enum": ["backup_and_reset"],
            },
            "transfer_on_respawn": {"type": "boolean"},
        }, required=(
            "namespace",
            "schema_version",
            "migrations",
            "malformed_policy",
            "transfer_on_respawn",
        ))
    if kind == "network_sync":
        return _schema({
            "sync_interval_ticks": {
                "type": "integer",
                "minimum": 1,
                "maximum": 1200,
            },
            "max_payload_bytes": {
                "type": "integer",
                "minimum": 256,
                "maximum": 32767,
            },
        }, required=(
            "sync_interval_ticks",
            "max_payload_bytes",
        ))
    if kind == "resource_policy":
        return _schema({})
    if kind == "quest":
        return _schema({
            "objective": {"type": "string", "enum": ["kill", "break", "manual"]},
            "target": _STRING,
            "required": _POSITIVE_INT,
            "reward_item": _RESOURCE_ID,
            "reward_count": _POSITIVE_INT,
            "reward_currency": {"type": "number", "minimum": 0},
        }, required=(
            "objective",
            "target",
            "required",
            "reward_item",
            "reward_count",
            "reward_currency",
        ))
    if kind == "class":
        return _schema({"display_name": _DISPLAY}, required=("display_name",))
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
        }, required=(
            "effect",
            "duration_ticks",
            "amplifier",
            "cooldown_ticks",
        ))
    if kind == "economy":
        return _schema(
            {"initial_balance": {"type": "number", "minimum": 0}},
            required=("initial_balance",),
        )
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
                }, required=("id", "item", "count", "price")),
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
                }, required=("slot", "item", "count")),
            },
        }, required=("template", "title", "rows", "entries"))
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
                }, required=("id", "type", "item", "count")),
                _schema({
                    "id": {"type": "string", "pattern": r"^[a-z][a-z0-9_]{1,63}$"},
                    "type": {"const": "status_effect"},
                    "effect": _RESOURCE_ID,
                    "duration_ticks": _POSITIVE_INT,
                    "amplifier": {"type": "integer", "minimum": 0, "maximum": 255},
                }, required=("id", "type", "effect", "duration_ticks", "amplifier")),
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
        return _schema({"display_name": _DISPLAY}, required=("display_name",))
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


def _coverage_allowed(kind: str, cover: str) -> bool:
    host_owner = host_platform_kind_for_ref(cover)
    if host_owner is not None:
        return kind == host_owner

    if cover.startswith("resources_and_ui."):
        concern = cover.split(".", 1)[1]
        registry_kinds = {
            "item", "block", "tool", "weapon", "armor", "food", "crop",
            "machine", "effect", "enchantment",
        }
        asset_kinds = {
            "item", "block", "tool", "weapon", "armor", "food", "crop",
            "machine",
        }
        if concern == "registries":
            return (
                kind in registry_kinds
                or kind in PLATFORM_SYSTEM_KIND_TO_PACK
                or kind in PLATFORM_ENTITY_KINDS
            )
        if concern == "data_resources":
            return (
                kind in PLATFORM_CONTENT_KINDS
                or kind in PLATFORM_SYSTEM_KIND_TO_PACK
                or kind in PLATFORM_ENTITY_KINDS
            )
        if concern == "assets":
            return kind in asset_kinds or kind in PLATFORM_ENTITY_KINDS
        if concern == "paths":
            return (
                kind in PLATFORM_CONTENT_KINDS
                or kind in PLATFORM_SYSTEM_KIND_TO_PACK
                or kind in PLATFORM_ENTITY_KINDS
            )
        if concern == "interactions":
            return kind in {"machine", "gui", "networking"}
        if concern == "displayed_state":
            return kind == "gui"
        return False

    if cover.startswith("integration."):
        return cover in {
            "integration.responsibilities",
            "integration.target_bindings",
            "integration.initialization_order",
            "integration.module_interfaces",
            "integration.side_placement",
            "integration.compatibility",
        }
    return False


def platform_coverable_refs(
    kind: str,
    refs: Sequence[str],
) -> tuple[str, ...]:
    if kind not in PLATFORM_KINDS:
        raise ValueError(f"TYPED_PLATFORM_KIND_UNSUPPORTED: {kind!r}")
    return tuple(
        ref
        for ref in refs
        if isinstance(ref, str) and _coverage_allowed(kind, ref)
    )


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
        "tag": {"registry", "values", "replace"},
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
    if kind == "tag":
        registry = config.get("registry")
        if registry not in {"items", "blocks", "entity_types", "fluids", "functions"}:
            raise ValueError(
                f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.registry"
            )
        values = config.get("values")
        if (
            not isinstance(values, list)
            or not values
            or len(values) > 256
            or any(
                not isinstance(value, str) or not _RESOURCE.fullmatch(value)
                for value in values
            )
        ):
            raise ValueError(
                f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.values"
            )
        if "replace" in config and type(config["replace"]) is not bool:
            raise ValueError(
                f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.replace"
            )


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
            normalized_cover = cover.strip()
            if not _coverage_allowed(kind, normalized_cover):
                raise ValueError(
                    f"{where}.covers: {kind!r} cannot implement "
                    f"{normalized_cover!r}"
                )
            normalized_covers.append(normalized_cover)

        seen.add(module_id)
        record = {
            "module_id": module_id,
            "kind": kind,
            "config": deepcopy(dict(config)),
            "covers": list(dict.fromkeys(normalized_covers)),
        }
        if kind in PLATFORM_CONTENT_KINDS:
            _validate_content_config(kind, config, module_id)
        elif kind in PLATFORM_ENTITY_KINDS:
            required = {
                "max_health",
                "attack_damage",
                "movement_speed",
                "follow_range",
                "archetype",
                "behavior",
                "entity_width",
                "entity_height",
                "spawn_group",
                "main_color",
            }
            optional = {"texture_width", "texture_height"}
            if not required <= set(config) or set(config) - required - optional:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id} entity fields"
                )
            for field in {
                "max_health",
                "attack_damage",
                "movement_speed",
                "follow_range",
                "entity_width",
                "entity_height",
            }:
                value = config[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.{field}"
                    )
            if float(config["max_health"]) <= 0:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.max_health"
                )
            if float(config["movement_speed"]) <= 0:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.movement_speed"
                )
            if float(config["follow_range"]) <= 0:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.follow_range"
                )
            if float(config["entity_width"]) <= 0 or float(config["entity_height"]) <= 0:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.entity_size"
                )
            if float(config["attack_damage"]) < 0:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.attack_damage"
                )
            if config["behavior"] in {"hostile_melee", "neutral_melee"} and float(
                config["attack_damage"]
            ) <= 0:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.attack_damage"
                )
            if config["archetype"] not in {
                "biped", "quadruped", "flying", "serpentine", "construct"
            }:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.archetype"
                )
            if config["behavior"] not in {
                "hostile_melee", "neutral_melee", "passive", "npc"
            }:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.behavior"
                )
            if config["spawn_group"] not in {
                "monster", "creature", "ambient", "water_creature", "misc"
            }:
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.spawn_group"
                )
            if not _HEX.fullmatch(str(config["main_color"])):
                raise ValueError(
                    f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.main_color"
                )
            for field in ("texture_width", "texture_height"):
                if field in config and (
                    type(config[field]) is not int
                    or not 1 <= config[field] <= 4096
                ):
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.{field}"
                    )
        elif kind in PLATFORM_HOST_KINDS:
            if kind == "state_store":
                allowed = {
                    "namespace",
                    "schema_version",
                    "migrations",
                    "malformed_policy",
                    "transfer_on_respawn",
                }
                unknown = set(config) - allowed
                if unknown:
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id} has "
                        f"unsupported fields {sorted(unknown)}"
                    )
                namespace = str(config.get("namespace", "authored_state"))
                if not re.fullmatch(r"[a-z][a-z0-9_.-]{1,63}", namespace):
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.namespace"
                    )
                schema_version = str(config.get("schema_version", "1")).strip()
                if not schema_version or len(schema_version) > 256:
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.schema_version"
                    )
                migrations = config.get("migrations", [])
                if not isinstance(migrations, list) or len(migrations) > 64:
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.migrations"
                    )
                for migration in migrations:
                    if not isinstance(migration, Mapping):
                        raise ValueError(
                            f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.migrations"
                        )
                    required = {"from_version", "to_version", "operation"}
                    optional = {"source_key", "destination_key", "value"}
                    if not required <= set(migration) or set(migration) - required - optional:
                        raise ValueError(
                            f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.migration"
                        )
                    operation = str(migration["operation"])
                    if operation not in {
                        "preserve",
                        "rename_key",
                        "delete_key",
                        "set_default",
                    }:
                        raise ValueError(
                            f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.migration.operation"
                        )
                    if operation in {"rename_key", "delete_key"} and not str(
                        migration.get("source_key") or ""
                    ).strip():
                        raise ValueError(
                            f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.migration.source_key"
                        )
                    if operation == "rename_key" and not str(
                        migration.get("destination_key") or ""
                    ).strip():
                        raise ValueError(
                            f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.migration.destination_key"
                        )
                    if operation == "set_default" and not str(
                        migration.get("destination_key") or ""
                    ).strip():
                        raise ValueError(
                            f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.migration.destination_key"
                        )
                if config.get("malformed_policy", "backup_and_reset") != "backup_and_reset":
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.malformed_policy"
                    )
                if "transfer_on_respawn" in config and type(
                    config["transfer_on_respawn"]
                ) is not bool:
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.transfer_on_respawn"
                    )
                cover_set = set(normalized_covers)
                if "persistence.migration" in cover_set and not migrations:
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id} migration "
                        "coverage requires typed migrations"
                    )
                if (
                    "persistence.transfers" in cover_set
                    and config.get("transfer_on_respawn") is not True
                ):
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id} transfers "
                        "coverage requires transfer_on_respawn=true"
                    )
            elif kind == "network_sync":
                allowed = {"sync_interval_ticks", "max_payload_bytes"}
                unknown = set(config) - allowed
                if unknown:
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id} has "
                        f"unsupported fields {sorted(unknown)}"
                    )
                for field, lower, upper, default in (
                    ("sync_interval_ticks", 1, 1200, 20),
                    ("max_payload_bytes", 256, 32767, 32767),
                ):
                    value = config.get(field, default)
                    if type(value) is not int or not lower <= value <= upper:
                        raise ValueError(
                            f"TYPED_PLATFORM_CONFIG_INVALID: {module_id}.{field}"
                        )
            elif kind == "resource_policy":
                if config:
                    raise ValueError(
                        f"TYPED_PLATFORM_CONFIG_INVALID: {module_id} resource_policy "
                        "takes no model-authored configuration"
                    )
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
    "NETWORK_SYNC_STATEFUL_REFS",
    "PLATFORM_CONTENT_KINDS",
    "PLATFORM_ENTITY_KINDS",
    "PLATFORM_HOST_KINDS",
    "PLATFORM_HOST_MODULE_IDS",
    "PLATFORM_HOST_SECTION_OWNER",
    "PLATFORM_KINDS",
    "PLATFORM_SYSTEM_KIND_TO_PACK",
    "host_platform_kind_for_ref",
    "network_sync_requires_state",
    "platform_config_schema",
    "platform_coverable_refs",
    "platform_module_authoring_schema",
    "validate_platform_modules",
]
