"""Migrate the pinned published HOST catalog to emit modern BlockItem client models.

The archived HOST catalog already contains the reviewed client_block_item
template, but its minecraft/block/model binding omits it. Merely updating the
rules builder does not change a serialized HOST snapshot. Repair exactly this
missing binding at catalog publication, and rebind immutable context digests.
Unknown template revisions or binding shapes remain fatal.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256

from packaging.version import Version

from .resolved_version_context import ResolvedVersionContext, _encode

_BLOCK_LEAF = "minecraft/block/model"
_BLOCK_MODEL = "minecraft/resource/block/model_cube_all"
_BLOCK_ITEM_MODEL = "minecraft/resource/item/client_block_item"


def migrate_published_block_item_authority(catalog: dict) -> dict:
    """Attach the registered modern BlockItem model to exact HOST block bindings."""
    from .task_template_catalog import load_template

    template_hash = "sha256:" + sha256(
        _encode(load_template(_BLOCK_ITEM_MODEL)).encode("utf-8")
    ).hexdigest()
    if template_hash != (
        "sha256:a70cf3f45b565bd8029c24a1e39d005181096c342c913bbca6e3bffc0c81292c"
    ):
        raise ValueError("HOST_BLOCK_ITEM_TEMPLATE_REVISION_UNREVIEWED")

    original_auto = catalog["auto_context_id"]
    remap: dict[str, str] = {}
    result = deepcopy(catalog)
    for bundle in result["bundles"]:
        if not isinstance(bundle, dict):
            raise ValueError("HOST_BLOCK_ITEM_CATALOG_INVALID_SHAPE")
        target = bundle.get("target")
        facts = bundle.get("host_facts")
        if not isinstance(target, dict) or not isinstance(facts, dict):
            raise ValueError("HOST_BLOCK_ITEM_CATALOG_INVALID_SHAPE")

        old_id = bundle.get("context_id")
        version = Version(str(target["minecraft_version"]))
        if version < Version("1.21.4"):
            if isinstance(old_id, str):
                remap[old_id] = old_id
            continue

        bindings = facts.get("leaf_bindings")
        rules = facts.get("artifact_rules")
        if not isinstance(bindings, dict) or not isinstance(rules, dict):
            raise ValueError("HOST_BLOCK_ITEM_CATALOG_INVALID_SHAPE")
        # Allow small unrelated HOST fixtures with no block-model scope.
        if _BLOCK_LEAF not in bindings and _BLOCK_ITEM_MODEL not in rules:
            remap[old_id] = old_id
            continue

        row = bindings.get(_BLOCK_LEAF)
        impl = row.get("implementation") if isinstance(row, dict) else None
        rule = rules.get(_BLOCK_ITEM_MODEL)
        if (
            not isinstance(impl, dict)
            or impl.get("implementation_id") != _BLOCK_MODEL
            or impl.get("template") != _BLOCK_MODEL
            or not isinstance(rule, dict)
            or rule.get("template_sha256") != template_hash
        ):
            raise ValueError("HOST_BLOCK_ITEM_BINDING_UNRECOGNIZED:" + str(version))
        extras = impl.get("extra_templates", [])
        if extras == [_BLOCK_ITEM_MODEL]:
            remap[old_id] = old_id
            continue
        if extras not in ([], None):
            raise ValueError("HOST_BLOCK_ITEM_BINDING_UNRECOGNIZED:" + str(version))
        impl["extra_templates"] = [_BLOCK_ITEM_MODEL]
        snapshot = {
            "target": target,
            "host_facts": facts,
            "source": bundle["source"],
        }
        new_context = ResolvedVersionContext(_encode(snapshot))
        bundle["context_id"] = new_context.context_id
        remap[old_id] = new_context.context_id

    if original_auto not in remap:
        raise ValueError("HOST_BLOCK_ITEM_AUTO_CONTEXT_MISSING")
    result["auto_context_id"] = remap[original_auto]
    return result


__all__ = ["migrate_published_block_item_authority"]
