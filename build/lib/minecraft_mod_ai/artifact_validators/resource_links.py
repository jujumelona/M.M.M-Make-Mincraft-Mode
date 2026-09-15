"""Validate resource references against produced or host-provided registry identities."""

import json


def validate_resource_links(rendered, values, registry):
    data = json.loads(rendered)
    kind = values.get("registry_kind", "item")
    if "values" in data:
        references = data["values"]
    elif data.get("type") == "minecraft:crafting_shaped":
        references = list(data["key"].values()) + [data["result"]["id"]]
    elif data.get("type") == "minecraft:crafting_shapeless":
        references = data["ingredients"] + [data["result"]["id"]]
    else:
        references = [data["ingredient"], data["result"]["id"]]
    expected_type = {"item": "Item", "block": "Block", "entity_type": "EntityType<?>"}[
        kind
    ]
    available = set(values.get("known_registry_ids", {}).get(kind, ()))
    if registry is not None:
        available.update(
            p.value
            for p in registry.all_ports().values()
            if p.port_kind.value == "REGISTRY_ID" and p.target_type == expected_type
        )
    missing = set(references) - available
    if missing:
        raise ValueError(f"RESOURCE_REFERENCE_MISSING: {sorted(missing)}")
    return {
        "validator": "resource_references",
        "status": "PASS",
        "references": references,
    }
