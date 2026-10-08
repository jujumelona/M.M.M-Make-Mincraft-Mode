"""Strict HOST snapshot migration for the previously misbound entity loot leaf.

The packaged version catalog is immutable input.  The retired entity binding
incorrectly pointed at fabric/loot/block_drop; the corresponding block template
also carried the pre-1.21 loot path.  Until the large published catalog is
regenerated as a whole, migrate ONLY that exact known snapshot shape with
registered template identities and recompute coherent immutable context IDs.

This is not speculative target discovery or a model-side template override.
"""
from __future__ import annotations

from hashlib import sha256

from .resolved_version_context import ResolvedVersionContext, _encode

_LEGACY_BLOCK_TEMPLATE_HASH = (
    "sha256:0fad571db62a77033911e9da32f696e13a40c5085c1b15cfb5d054c40843cf01"
)
_LEGACY_BLOCK_FILE_HASH = (
    "sha256:12e93bb3b42152002c1440884ea77cba00bf60b56a9831278480a42880311355"
)
_BLOCK = "fabric/loot/block_drop"
_ENTITY = "fabric/loot/entity_drop"


def migrate_published_loot_authority(catalog: dict) -> dict:
    """Upgrade only the known misbound loot contract; reject unexpected inputs."""
    from .populate_version_artifact_rules import (
        TEMPLATE_REQUIREMENTS,
        make_implementation,
    )
    from .task_template_catalog import load_template

    template_hashes = {
        identifier: "sha256:" + sha256(_encode(load_template(identifier)).encode()).hexdigest()
        for identifier in (_BLOCK, _ENTITY)
    }
    if template_hashes[_BLOCK] != (
        "sha256:82d1826b61ca3b1e873a3b9525d7b3b259688ec14b8bc253d16e51275a769d3c"
    ) or template_hashes[_ENTITY] != (
        "sha256:3b91f91e17277f400e3f586d3e96de7002e720849325032309f0ff7b98d14f1f"
    ):
        raise ValueError("HOST_LOOT_TEMPLATE_REVISION_UNREVIEWED")

    # A freshly republished HOST catalog already contains the new bindings.
    # Accept that coherent state unchanged; mixed/unknown snapshots remain
    # rejected by the pinned legacy migration below.
    already_current = all(
        bundle["host_facts"]["artifact_rules"].get(_BLOCK, {}).get(
            "template_sha256"
        ) == template_hashes[_BLOCK]
        and bundle["host_facts"]["artifact_rules"].get(_ENTITY, {}).get(
            "template_sha256"
        ) == template_hashes[_ENTITY]
        and bundle["host_facts"]["leaf_bindings"]["minecraft/loot/entry"]
        .get("implementation", {}).get("implementation_id") == _ENTITY
        for bundle in catalog["bundles"]
    )
    if already_current:
        return catalog

    auto_before = catalog["auto_context_id"]
    remap: dict[str, str] = {}
    for bundle in catalog["bundles"]:
        facts = bundle["host_facts"]
        rules = facts["artifact_rules"]
        block_binding = facts["leaf_bindings"]["minecraft/block/drops"]
        entity_binding = facts["leaf_bindings"]["minecraft/loot/entry"]
        old_block = rules.get(_BLOCK)
        old_block_impl = block_binding.get("implementation", {})
        old_entity_impl = entity_binding.get("implementation", {})
        if (
            not isinstance(old_block, dict)
            or old_block.get("template_sha256") != _LEGACY_BLOCK_TEMPLATE_HASH
            or old_block_impl.get("implementation_id") != _BLOCK
            or old_block_impl.get("implementation_sha256") != _LEGACY_BLOCK_FILE_HASH
            or old_entity_impl.get("implementation_id") != _BLOCK
            or old_entity_impl.get("implementation_sha256") != _LEGACY_BLOCK_FILE_HASH
            or _ENTITY in rules
            or facts["capabilities"].get("LOOT_TABLE") is not True
        ):
            raise ValueError(
                "HOST_LOOT_CATALOG_MIGRATION_UNRECOGNIZED: "
                + str(bundle["target"]["minecraft_version"])
            )

        version = bundle["target"]["minecraft_version"]
        rules[_BLOCK]["template_sha256"] = template_hashes[_BLOCK]
        entity_requirements = TEMPLATE_REQUIREMENTS[_ENTITY]
        rules[_ENTITY] = {
            "template_sha256": template_hashes[_ENTITY],
            "requires_capabilities": list(entity_requirements["requires_capabilities"]),
            "required_symbols": list(entity_requirements["required_symbols"]),
        }
        block_binding["implementation"] = make_implementation(
            "minecraft/block/drops", version,
            template_id=_BLOCK, executor_type="deterministic_renderer",
            validator_profile="json_schema",
        )
        entity_binding["implementation"] = make_implementation(
            "minecraft/loot/entry", version,
            template_id=_ENTITY, executor_type="deterministic_renderer",
            validator_profile="json_schema",
        )
        old_context_id = bundle["context_id"]
        snapshot = {
            "target": bundle["target"],
            "host_facts": facts,
            "source": bundle["source"],
        }
        # Instantiate the real contract rather than setting a forged digest.
        new_context = ResolvedVersionContext(_encode(snapshot))
        bundle["context_id"] = new_context.context_id
        remap[old_context_id] = new_context.context_id

    if auto_before not in remap:
        raise ValueError("HOST_LOOT_AUTO_CONTEXT_MISSING")
    catalog["auto_context_id"] = remap[auto_before]
    return catalog


__all__ = ["migrate_published_loot_authority"]
