"""Publish reviewed HOST artifact rules on top of the migrating core builder.

Target-specific facts must be normalized here (or in focused HOST authority
modules) before they are written to the catalog. The core module is retained
only while remaining leaf families are migrated away from legacy inference.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

from . import _version_artifact_rules_core as _core
from .host_item_registration import item_registration_epoch
from .resolved_version_context import ResolvedVersionContext, _encode
from .task_template_catalog import load_template

DEFAULT_DATA_DIR = Path(__file__).with_name("data")

ITEM_EPOCH_TEMPLATES = (
    "fabric/item/register_direct_resource_location_ctor",
    "fabric/item/register_direct_resource_location_factory",
    "fabric/item/key_resource_location",
    "fabric/item/key_identifier",
    "fabric/item/register_keyed",
)
# Templates selected directly by HOST code rather than YAML sequence edges.  Template
# reachability validation must count these as real code consumers, but deliberately
# excludes legacy templates so they can become detectable/deletable once core use ends.
HOST_TEMPLATE_CANDIDATES = ITEM_EPOCH_TEMPLATES
LEAF_TEMPLATES = tuple(dict.fromkeys((*_core.LEAF_TEMPLATES, *ITEM_EPOCH_TEMPLATES)))
ARTIFACT_SCHEMAS = _core.ARTIFACT_SCHEMAS
TEMPLATE_REQUIREMENTS = dict(_core.TEMPLATE_REQUIREMENTS)
TEMPLATE_REQUIREMENTS.update(
    {
        "fabric/item/register_direct_resource_location_ctor": {
            "requires_capabilities": ["REGISTER_ITEM"],
            "required_symbols": ["register_item", "identifier_factory"],
        },
        "fabric/item/register_direct_resource_location_factory": {
            "requires_capabilities": ["REGISTER_ITEM"],
            "required_symbols": ["register_item", "identifier_factory"],
        },
        "fabric/item/key_resource_location": {
            "requires_capabilities": ["REGISTER_ITEM"],
            "required_symbols": ["resource_key_create", "identifier_factory"],
        },
        "fabric/item/key_identifier": {
            "requires_capabilities": ["REGISTER_ITEM"],
            "required_symbols": ["resource_key_create", "identifier_factory"],
        },
        "fabric/item/register_keyed": {
            "requires_capabilities": ["REGISTER_ITEM"],
            "required_symbols": ["register_item"],
        },
    }
)

make_implementation = _core.make_implementation
all_canonical_leaves = _core.all_canonical_leaves
_sha256_text = _core._sha256_text


def template_hashes() -> dict[str, str]:
    """Return hashes for legacy core templates plus reviewed epoch templates."""
    from .integrity_bootstrap import bootstrap_integrity

    authority = bootstrap_integrity()
    authority.verify_live()
    for identifier in LEAF_TEMPLATES:
        authority.implementations.get_implementation(identifier)
    return {identifier: _sha256_text(_encode(load_template(identifier))) for identifier in LEAF_TEMPLATES}


def _item_template_ids(minecraft_version: str) -> tuple[str, str | None]:
    epoch = item_registration_epoch(minecraft_version)
    if epoch["id"] == "direct_resource_location":
        registration = (
            "fabric/item/register_direct_resource_location_factory"
            if epoch["resource_identifier_factory_static"]
            else "fabric/item/register_direct_resource_location_ctor"
        )
        return registration, None
    if epoch["id"] == "keyed_resource_location":
        return "fabric/item/register_keyed", "fabric/item/key_resource_location"
    if epoch["id"] == "keyed_identifier":
        return "fabric/item/register_keyed", "fabric/item/key_identifier"
    raise ValueError(f"HOST_ITEM_REGISTRATION_EPOCH_UNSUPPORTED:{epoch['id']}")


def _symbol(
    *, owner: str, name: str, descriptor: str, kind: str = "method", static: bool = True
) -> dict[str, object]:
    return {
        "owner": owner,
        "name": name,
        "descriptor": descriptor,
        "kind": kind,
        "static": static,
        "side": "common",
        "namespace": "minecraft",
    }


def _apply_item_host_authority(
    facts: dict, minecraft_version: str, hashes: dict[str, str]
) -> dict:
    """Replace legacy item guesses with one reviewed Mojang-mapped HOST row."""
    epoch = item_registration_epoch(minecraft_version)
    registration_template, key_template = _item_template_ids(minecraft_version)

    rules = facts["artifact_rules"]
    rules.pop("fabric/item/key", None)
    rules.pop("fabric/item/register_basic", None)
    selected = [registration_template]
    if key_template:
        selected.append(key_template)
    for template_id in selected:
        req = TEMPLATE_REQUIREMENTS[template_id]
        rules[template_id] = {
            "template_sha256": hashes[template_id],
            "requires_capabilities": list(req["requires_capabilities"]),
            "required_symbols": list(req["required_symbols"]),
        }

    extra = {"prerequisite_templates": [key_template]} if key_template else None
    facts["leaf_bindings"]["minecraft/item/registry"] = {
        "state": "not_reviewed",
        "reason": "REAL_EXECUTION_EVIDENCE_REQUIRED",
        "implementation": make_implementation(
            "minecraft/item/registry",
            minecraft_version,
            template_id=registration_template,
            executor_type="deterministic_renderer",
            validator_profile="java_syntax",
            hashes=hashes,
            extra=extra,
        ),
    }

    identifier_owner = str(epoch["resource_identifier_owner"])
    identifier_internal = identifier_owner.replace(".", "/")
    identifier_factory = str(epoch["resource_identifier_factory"])
    identifier_static = bool(epoch["resource_identifier_factory_static"])
    resource_key_owner = str(epoch["resource_key_owner"])
    resource_key_internal = resource_key_owner.replace(".", "/")
    registry_owner = str(epoch["registry_owner"])

    register_id_type = resource_key_internal if epoch["requires_resource_key"] else identifier_internal
    facts["api_symbols"].update(
        {
            "register_item": _symbol(
                owner=registry_owner,
                name="register",
                descriptor=(
                    f"(Lnet/minecraft/core/Registry;L{register_id_type};Ljava/lang/Object;)Ljava/lang/Object;"
                ),
            ),
            "resource_key_create": _symbol(
                owner=resource_key_owner,
                name="create",
                descriptor=(
                    f"(Lnet/minecraft/resources/ResourceKey;L{identifier_internal};)"
                    "Lnet/minecraft/resources/ResourceKey;"
                ),
            ),
            "builtin_item_registry": _symbol(
                owner=str(epoch["builtin_registries_owner"]),
                name="ITEM",
                descriptor="Lnet/minecraft/core/DefaultedRegistry;",
                kind="field",
            ),
            "identifier_factory": _symbol(
                owner=identifier_owner,
                name=identifier_factory,
                descriptor=f"(Ljava/lang/String;Ljava/lang/String;)L{identifier_internal};",
                kind="method" if identifier_static else "constructor",
                static=identifier_static,
            ),
            "item_stacks_to": _symbol(
                owner="net.minecraft.world.item.Item$Properties",
                name="stacksTo",
                descriptor="(I)Lnet/minecraft/world/item/Item$Properties;",
                static=False,
            ),
            "registries_item": _symbol(
                owner=str(epoch["registries_owner"]),
                name="ITEM",
                descriptor="Lnet/minecraft/resources/ResourceKey;",
                kind="field",
            ),
        }
    )

    if epoch["id"] == "keyed_identifier":
        replacement = "Identifier.fromNamespaceAndPath"
    elif epoch["resource_identifier_factory_static"]:
        replacement = "ResourceLocation.fromNamespaceAndPath"
    else:
        replacement = "new ResourceLocation"
    facts.setdefault("replacements", {})["new Identifier"] = replacement
    return facts


def build_version_facts(
    minecraft_version: str,
    *,
    base_facts: dict,
    hashes: dict[str, str] | None = None,
) -> dict:
    if hashes is None:
        hashes = template_hashes()
    facts = _core.build_version_facts(
        minecraft_version,
        base_facts=base_facts,
        hashes=hashes,
    )
    return _apply_item_host_authority(facts, minecraft_version, hashes)


def populate_catalog(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[int, str]:
    """Publish a catalog using only the reviewed public HOST builder."""
    catalog_path = data_dir / "host_version_catalog.json"
    evidence_path = data_dir / "official_version_evidence.json"

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    evidence_report = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence_by_mc = {row["minecraft"]: row for row in evidence_report["versions"]}

    hashes = template_hashes()
    new_bundles = []
    new_evidence_rows = []

    for bundle in catalog["bundles"]:
        minecraft = bundle["target"]["minecraft_version"]
        facts = build_version_facts(minecraft, base_facts=bundle["host_facts"], hashes=hashes)

        old_row = dict(evidence_by_mc[minecraft])
        row = dict(old_row)
        row.pop("context_id", None)
        row["artifact_rules_count"] = len(facts["artifact_rules"])
        row["capabilities_count"] = len(facts["capabilities"])
        row["unverified"] = ["runtime"]

        rev_digest = sha256(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        facts["host_revision"] = "sha256:" + rev_digest

        raw = {"target": bundle["target"], "host_facts": facts, "source": "HOST"}
        ctx = ResolvedVersionContext(_encode(raw))
        row["context_id"] = ctx.context_id
        new_bundles.append(ctx.to_dict())
        new_evidence_rows.append(row)

    auto_context_id = new_bundles[0]["context_id"]
    catalog["auto_context_id"] = auto_context_id
    catalog["bundles"] = new_bundles
    evidence_report["versions"] = new_evidence_rows

    catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence_path.write_text(
        json.dumps(evidence_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(new_bundles), auto_context_id


def __getattr__(name: str):
    """Compatibility bridge for non-migrated core helpers during the split."""
    return getattr(_core, name)


if __name__ == "__main__":
    count, auto_id = populate_catalog()
    print(f"Populated {count} bundles. AUTO context_id: {auto_id}")
