from __future__ import annotations

"""Single source of truth for reviewed deterministic production backends.

Semantic module kinds are not executable capabilities by themselves.  This module
maps each built-in production route to the exact provider receipt capabilities that
must be present before that route may be planned or dispatched.
"""

import json
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
    "block_entity",
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

LOCAL_AI_SIDECAR_INTEGRATION_TYPE = "mmm_local_ai_sidecar"
RESEARCH_SHARD_INTEGRATION_TYPE = "mmm_research_shard"

NATIVE_PRODUCTION_STAGE_BY_KIND = {
    **{kind: "content" for kind in EXTENDED_CONTENT_KINDS},
    **{kind: "system" for kind in SYSTEM_KIND_TO_PACK},
    **{kind: "entity" for kind in ENTITY_PIPELINE_KINDS},
    "typed_host": "host",
}
NATIVE_INTEGRATION_STAGE_BY_TYPE = {
    LOCAL_AI_SIDECAR_INTEGRATION_TYPE: "content",
    RESEARCH_SHARD_INTEGRATION_TYPE: "content",
}
NATIVE_PRODUCTION_MODULE_KINDS = frozenset(NATIVE_PRODUCTION_STAGE_BY_KIND)
NATIVE_INTEGRATION_TYPES = frozenset(NATIVE_INTEGRATION_STAGE_BY_TYPE)


def native_production_stage(
    kind: str,
    config: Mapping[str, Any] | None = None,
) -> str | None:
    normalized = str(kind or "").strip()
    stage = NATIVE_PRODUCTION_STAGE_BY_KIND.get(normalized)
    if stage is not None:
        return stage
    if normalized != "integration":
        return None
    integration_type = str(
        (config or {}).get("integration_type") or ""
    ).strip()
    return NATIVE_INTEGRATION_STAGE_BY_TYPE.get(integration_type)


def native_production_route_available(
    kind: str,
    config: Mapping[str, Any] | None = None,
) -> bool:
    return native_production_stage(kind, config) is not None


def deterministic_backend_capabilities(target: Any) -> frozenset[str]:
    """Return only provider-reviewed fixed-generator backend capabilities.

    This deliberately does not infer support from host_facts.capabilities. Host facts
    describe API/leaf availability and are consumed by artifact/template admission;
    they do not prove that a legacy fixed generator was reviewed for the target.
    """

    if isinstance(target, Mapping):
        values = target.get("deterministic_module_kinds", ())
    else:
        values = getattr(target, "deterministic_module_kinds", ())
    return normalize_capabilities(values)

def target_semantic_capabilities(target: Any) -> frozenset[str]:
    """Return the union of provider backend tokens and immutable host capability facts.

    Use this for semantic/API capability reasoning only. Fixed deterministic generator
    admission must use deterministic_backend_capabilities() so host API facts cannot
    accidentally authorize an unreviewed generator.
    """

    if isinstance(target, Mapping):
        raw_deterministic = target.get("deterministic_module_kinds", ())
        host_facts_json = target.get("host_facts_json", "")
    else:
        raw_deterministic = getattr(target, "deterministic_module_kinds", ())
        host_facts_json = getattr(target, "host_facts_json", "")

    result = set(normalize_capabilities(raw_deterministic))
    raw_host_facts = str(host_facts_json or "").strip()
    if not raw_host_facts:
        return frozenset(result)

    try:
        host_facts = json.loads(raw_host_facts)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("TARGET_HOST_FACTS_INVALID: host_facts_json is not valid JSON") from exc
    if not isinstance(host_facts, Mapping):
        raise ValueError("TARGET_HOST_FACTS_INVALID: host_facts_json must contain an object")

    capabilities = host_facts.get("capabilities", {})
    if not isinstance(capabilities, Mapping):
        raise ValueError("TARGET_HOST_FACTS_INVALID: capabilities must be an object")
    for name, supported in capabilities.items():
        if type(supported) is not bool:
            raise ValueError(
                f"TARGET_HOST_FACTS_INVALID: capability {name!r} must be boolean"
            )
        if supported and str(name).strip():
            result.add(str(name).strip())
    return frozenset(result)

def effective_target_backend_capabilities(target: Any) -> frozenset[str]:
    """Compatibility name with fail-closed backend semantics.

    Historical callers used this ambiguous name for both host API facts and fixed
    generator admission. It now deliberately returns only reviewed deterministic
    backend tokens. Use target_semantic_capabilities() for host-fact reasoning.
    """

    return deterministic_backend_capabilities(target)

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


def bootstrap_content_capabilities(
    kind: str,
    *,
    recipe: bool,
) -> frozenset[str]:
    """Capabilities consumed by the legacy bootstrap ContentSpec compiler."""

    normalized = str(kind or "").strip()
    if normalized == "item":
        required = {"item"}
        if recipe:
            required.add("recipe")
        return frozenset(required)
    if normalized == "block":
        required = {"block", "loot", "tag"}
        if recipe:
            required.add("recipe")
        return frozenset(required)
    return frozenset()


def bootstrap_boss_capabilities() -> frozenset[str]:
    """Capabilities consumed by FabricProjectGenerator._write_boss()."""

    return frozenset({"boss", "item", "loot"})


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
    "deterministic_backend_capabilities",
    "bootstrap_boss_capabilities",
    "bootstrap_content_capabilities",
    "ENTITY_PIPELINE_KINDS",
    "effective_target_backend_capabilities",
    "LOCAL_AI_SIDECAR_INTEGRATION_TYPE",
    "NATIVE_INTEGRATION_STAGE_BY_TYPE",
    "NATIVE_INTEGRATION_TYPES",
    "NATIVE_PRODUCTION_MODULE_KINDS",
    "NATIVE_PRODUCTION_STAGE_BY_KIND",
    "RESEARCH_SHARD_INTEGRATION_TYPE",
    "EXTENDED_CONTENT_KINDS",
    "SYSTEM_KIND_TO_PACK",
    "SYSTEM_PACK_KINDS",
    "geckolib_entity_capabilities",
    "missing_production_backend_capabilities",
    "native_production_route_available",
    "native_production_stage",
    "normalize_capabilities",
    "production_backend_is_supported",
    "production_module_backend_capabilities",
    "supported_extended_content_kinds",
    "system_pack_capabilities",
    "target_semantic_capabilities",
]
