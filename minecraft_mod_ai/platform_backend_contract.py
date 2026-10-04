from __future__ import annotations

"""Single source of truth for reviewed deterministic production backends.

Semantic module kinds are not executable capabilities by themselves.  This module
maps each built-in production route to the exact provider receipt capabilities that
must be present before that route may be planned or dispatched.
"""

from collections.abc import Iterable, Mapping
from typing import Any

EXTENDED_CONTENT_KINDS = frozenset({
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
    "tag",
})

SYSTEM_KIND_TO_PACK = {
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

SYSTEM_PACK_KINDS = {
    "quest-system": frozenset({"quest"}),
    "class-skill-system": frozenset({"class", "skill"}),
    "economy-shop": frozenset({"economy", "shop"}),
    "gui-networking": frozenset({"gui", "networking"}),
    "party-guild": frozenset({"party", "guild"}),
}

ENTITY_PIPELINE_KINDS = frozenset({"entity", "boss", "npc"})
DEFAULT_GECKOLIB_VERSION = "4.8.2"


def normalize_capabilities(values: Iterable[str] | None) -> frozenset[str]:
    if values is None:
        return frozenset()
    return frozenset(
        str(value).strip()
        for value in values
        if str(value).strip()
    )


def system_pack_capabilities(pack_id: str) -> frozenset[str]:
    normalized = str(pack_id or "").strip()
    semantic = SYSTEM_PACK_KINDS.get(normalized)
    if semantic is None:
        return frozenset()
    return frozenset((*semantic, f"system-pack:{normalized}"))


def geckolib_entity_capabilities(
    version: str = DEFAULT_GECKOLIB_VERSION,
) -> frozenset[str]:
    normalized = str(version or "").strip()
    if not normalized:
        return frozenset()
    return frozenset({
        "entity",
        "geckolib:entity",
        f"geckolib:version:{normalized}",
    })


def production_module_backend_capabilities(
    kind: str,
    config: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    """Return exact deterministic capabilities for the CompleteProduction route."""

    normalized = str(kind or "").strip()
    if normalized in EXTENDED_CONTENT_KINDS:
        return frozenset({normalized})

    pack_id = SYSTEM_KIND_TO_PACK.get(normalized)
    if pack_id is not None:
        return system_pack_capabilities(pack_id)

    if normalized in ENTITY_PIPELINE_KINDS:
        # CompleteProductionOrchestrator currently dispatches every entity/boss/npc
        # through generate_geckolib_entity_assets(), whose executable default is fixed
        # at DEFAULT_GECKOLIB_VERSION.  Do not infer a version from semantic config that
        # the dispatcher does not consume.
        return geckolib_entity_capabilities(DEFAULT_GECKOLIB_VERSION)

    return frozenset()


def missing_production_backend_capabilities(
    available: Iterable[str] | None,
    kind: str,
    config: Mapping[str, Any] | None = None,
) -> frozenset[str]:
    required = production_module_backend_capabilities(kind, config)
    if not required:
        return frozenset()
    return frozenset(required - normalize_capabilities(available))


def production_backend_is_supported(
    available: Iterable[str] | None,
    kind: str,
    config: Mapping[str, Any] | None = None,
) -> bool:
    required = production_module_backend_capabilities(kind, config)
    return bool(required) and not missing_production_backend_capabilities(
        available,
        kind,
        config,
    )


def supported_extended_content_kinds(
    available: Iterable[str] | None,
) -> frozenset[str]:
    capabilities = normalize_capabilities(available)
    return frozenset(
        kind
        for kind in EXTENDED_CONTENT_KINDS
        if kind in capabilities
    )


__all__ = [
    "DEFAULT_GECKOLIB_VERSION",
    "ENTITY_PIPELINE_KINDS",
    "EXTENDED_CONTENT_KINDS",
    "SYSTEM_KIND_TO_PACK",
    "SYSTEM_PACK_KINDS",
    "geckolib_entity_capabilities",
    "missing_production_backend_capabilities",
    "normalize_capabilities",
    "production_backend_is_supported",
    "production_module_backend_capabilities",
    "supported_extended_content_kinds",
    "system_pack_capabilities",
]
