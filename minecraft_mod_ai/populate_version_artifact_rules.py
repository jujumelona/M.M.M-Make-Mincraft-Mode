"""Populate verified artifact rules, capabilities, API symbols, schemas, and replacements across host version catalog bundles."""

from hashlib import sha256
import json
from pathlib import Path
from packaging.version import Version

from .resolved_version_context import ResolvedVersionContext, _encode
from .task_template_catalog import load_template

DEFAULT_DATA_DIR = Path(__file__).with_name("data")

LEAF_TEMPLATES = (
    "fabric/item/key",
    "fabric/item/register_basic",
    "fabric/item/client_item",
    "fabric/item/model_basic",
    "fabric/item/lang_en",
    "fabric/item/initializer",
    "fabric/item/settings_max_stack",
    "fabric/block/key",
    "fabric/block/register_basic",
    "fabric/block/blockstate_basic",
    "fabric/block/model_cube_all",
    "fabric/block/lang_en",
    "fabric/block/initializer",
    "fabric/recipe/shaped",
    "fabric/recipe/shapeless",
    "fabric/recipe/smelting",
    "fabric/tag/registry",
    "fabric/loot/block_drop",
)

ARTIFACT_SCHEMAS = {
    "fabric/item/client_item": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["model"],
        "properties": {
            "model": {
                "type": "object",
                "required": ["model", "type"],
                "properties": {
                    "model": {"type": "string"},
                    "type": {"type": "string"},
                },
            }
        },
    },
    "fabric/item/model_basic": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["parent", "textures"],
        "properties": {
            "parent": {"type": "string"},
            "textures": {
                "type": "object",
                "required": ["layer0"],
                "properties": {
                    "layer0": {"type": "string"},
                },
            },
        },
    },
    "fabric/item/lang_en": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
    },
    "fabric/block/blockstate_basic": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["variants"],
        "properties": {
            "variants": {
                "type": "object",
            },
        },
    },
    "fabric/block/model_cube_all": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["parent", "textures"],
        "properties": {
            "parent": {"type": "string"},
            "textures": {
                "type": "object",
                "required": ["all"],
                "properties": {
                    "all": {"type": "string"},
                },
            },
        },
    },
    "fabric/block/lang_en": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
    },
    "fabric/recipe/shaped": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["type", "pattern", "key", "result"],
        "properties": {
            "type": {"type": "string"},
            "pattern": {"type": "array", "items": {"type": "string"}},
            "key": {"type": "object"},
            "result": {"type": "object"},
        },
    },
    "fabric/recipe/shapeless": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["type", "ingredients", "result"],
        "properties": {
            "type": {"type": "string"},
            "ingredients": {"type": "array"},
            "result": {"type": "object"},
        },
    },
    "fabric/recipe/smelting": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["type", "ingredient", "result"],
        "properties": {
            "type": {"type": "string"},
            "ingredient": {"type": "object"},
            "result": {"type": "object"},
            "experience": {"type": ["number", "string"]},
            "cookingtime": {"type": ["integer", "string"]},
        },
    },
    "fabric/tag/registry": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["replace", "values"],
        "properties": {
            "replace": {"type": "boolean"},
            "values": {"type": "array", "items": {"type": "string"}},
        },
    },
    "fabric/loot/block_drop": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["type", "pools"],
        "properties": {
            "type": {"type": "string"},
            "pools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["rolls", "entries"],
                    "properties": {
                        "rolls": {"type": ["integer", "number"]},
                        "entries": {"type": "array"},
                    },
                },
            },
        },
    },
}


def template_hashes() -> dict[str, str]:
    hashes = {}
    for identifier in LEAF_TEMPLATES:
        template = load_template(identifier)
        digest = "sha256:" + sha256(_encode(template).encode()).hexdigest()
        hashes[identifier] = digest
    return hashes


from .structural_routing_contract import CANONICAL_ARTIFACT_KINDS
from .minecraft_template_steps import responsibility_ids_for_artifact


def all_canonical_leaves() -> tuple[str, ...]:
    leaves = []
    for kind in CANONICAL_ARTIFACT_KINDS:
        leaves.extend(responsibility_ids_for_artifact(kind))
    return tuple(leaves)


def build_version_facts(
    minecraft_version: str,
    *,
    base_facts: dict,
    hashes: dict[str, str] | None = None,
) -> dict:
    if hashes is None:
        hashes = template_hashes()
    v = Version(minecraft_version)
    is_modern_recipes = v >= Version("1.21.2")
    is_modern_registry = v >= Version("1.19.3")
    is_modern_id = v >= Version("1.21.0")

    admitted_templates = (
        list(LEAF_TEMPLATES)
        if is_modern_recipes
        else [
            t
            for t in LEAF_TEMPLATES
            if not t.startswith("fabric/recipe/") and t != "fabric/tag/registry"
        ]
    )

    rules = {}
    for tid in admitted_templates:
        if "item" in tid:
            req_cap = ["REGISTER_ITEM"]
        elif "block" in tid:
            req_cap = ["REGISTER_BLOCK"]
        elif "recipe" in tid:
            req_cap = ["RECIPE_CRAFTING"]
        elif "tag" in tid:
            req_cap = ["TAGS"]
        else:
            req_cap = ["LOOT_TABLE"]

        req_sym = []
        if tid == "fabric/item/register_basic":
            req_sym = ["register_item"]
        elif tid == "fabric/item/key":
            req_sym = ["resource_key_create"]
        elif tid == "fabric/item/settings_max_stack":
            req_sym = ["item_stacks_to"]
        elif tid == "fabric/block/register_basic":
            req_sym = ["register_item"]
        elif tid == "fabric/block/key":
            req_sym = ["resource_key_create"]

        rules[tid] = {
            "template_sha256": hashes[tid],
            "requires_capabilities": req_cap,
            "required_symbols": req_sym,
        }

    canonical_leaves = all_canonical_leaves()
    leaf_bindings = {}
    for leaf in canonical_leaves:
        if leaf == "minecraft/item/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "executor": "canonical_contract_validator",
                    "validator_profile": "semantic_contract",
                },
            }
        elif leaf == "minecraft/item/registry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/item/register_basic",
                    "prerequisite_templates": ["fabric/item/key"],
                    "template_sha256": hashes["fabric/item/register_basic"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "java_syntax",
                },
            }
        elif leaf == "minecraft/item/properties":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/item/settings_max_stack",
                    "template_sha256": hashes["fabric/item/settings_max_stack"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "java_syntax",
                },
            }
        elif leaf == "minecraft/item/model":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/item/model_basic",
                    "extra_templates": ["fabric/item/client_item"] if is_modern_recipes else [],
                    "template_sha256": hashes["fabric/item/model_basic"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "json_schema",
                },
            }
        elif leaf == "minecraft/item/language":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/item/lang_en",
                    "template_sha256": hashes["fabric/item/lang_en"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "json_schema",
                },
            }
        elif leaf == "minecraft/item/integration":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/item/initializer",
                    "template_sha256": hashes["fabric/item/initializer"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "java_syntax",
                },
            }
        elif leaf == "minecraft/item/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "executor": "canonical_contract_validator",
                    "validator_profile": "mod_integration_test",
                },
            }
        elif leaf == "minecraft/block/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "executor": "canonical_contract_validator",
                    "validator_profile": "semantic_contract",
                },
            }
        elif leaf == "minecraft/block/registry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/block/register_basic",
                    "prerequisite_templates": ["fabric/block/key"],
                    "template_sha256": hashes["fabric/block/register_basic"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "java_syntax",
                },
            }
        elif leaf == "minecraft/block/state":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/block/blockstate_basic",
                    "template_sha256": hashes["fabric/block/blockstate_basic"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "json_schema",
                },
            }
        elif leaf == "minecraft/block/model":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/block/model_cube_all",
                    "template_sha256": hashes["fabric/block/model_cube_all"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "json_schema",
                },
            }
        elif leaf == "minecraft/language/key":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/block/lang_en",
                    "template_sha256": hashes["fabric/block/lang_en"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "json_schema",
                },
            }
        elif leaf == "minecraft/block/integration":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/block/initializer",
                    "template_sha256": hashes["fabric/block/initializer"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "java_syntax",
                },
            }
        elif leaf == "minecraft/block/drops":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/loot/block_drop",
                    "template_sha256": hashes["fabric/loot/block_drop"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "json_schema",
                },
            }
        elif leaf == "minecraft/block/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "executor": "canonical_contract_validator",
                    "validator_profile": "mod_integration_test",
                },
            }
        elif leaf == "minecraft/recipe/requirement":
            if is_modern_recipes:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": {
                        "executor": "canonical_contract_validator",
                        "validator_profile": "semantic_contract",
                    },
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
                }
        elif leaf == "minecraft/recipe/serializer":
            if is_modern_recipes:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": {
                        "template": "fabric/recipe/shaped",
                        "supported_templates": [
                            "fabric/recipe/shaped",
                            "fabric/recipe/shapeless",
                            "fabric/recipe/smelting",
                        ],
                        "template_sha256": hashes["fabric/recipe/shaped"],
                        "executor": "deterministic_renderer",
                        "validator_profile": "json_schema",
                    },
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
                }
        elif leaf == "minecraft/recipe/validation":
            if is_modern_recipes:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": {
                        "executor": "canonical_contract_validator",
                        "validator_profile": "json_schema",
                    },
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
                }
        elif leaf == "minecraft/tag/requirement":
            if is_modern_recipes:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": {
                        "executor": "canonical_contract_validator",
                        "validator_profile": "semantic_contract",
                    },
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
                }
        elif leaf == "minecraft/tag/entries":
            if is_modern_recipes:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": {
                        "template": "fabric/tag/registry",
                        "template_sha256": hashes["fabric/tag/registry"],
                        "executor": "deterministic_renderer",
                        "validator_profile": "json_schema",
                    },
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
                }
        elif leaf == "minecraft/tag/validation":
            if is_modern_recipes:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": {
                        "executor": "canonical_contract_validator",
                        "validator_profile": "json_schema",
                    },
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
                }
        elif leaf == "minecraft/loot/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "executor": "canonical_contract_validator",
                    "validator_profile": "semantic_contract",
                },
            }
        elif leaf == "minecraft/loot/entry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "template": "fabric/loot/block_drop",
                    "template_sha256": hashes["fabric/loot/block_drop"],
                    "executor": "deterministic_renderer",
                    "validator_profile": "json_schema",
                },
            }
        elif leaf == "minecraft/loot/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "executor": "canonical_contract_validator",
                    "validator_profile": "json_schema",
                },
            }
        elif leaf in {
            "minecraft/entity/registry",
            "minecraft/screen/registration",
            "minecraft/network_payload/registration",
            "minecraft/block_entity/registry",
            "minecraft/effect/registry",
            "minecraft/sound/registration",
            "minecraft/particle/registry",
            "minecraft/item/interaction",
            "minecraft/block/interaction",
        }:
            suffix = leaf.split("/")[1]
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "executor": f"generator_handoff_{suffix}",
                    "canonical_leaf": leaf,
                    "validator_profile": "java_syntax",
                },
            }
        elif leaf == "minecraft/component/type":
            if v < Version("1.20.5"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "DATA_COMPONENTS_INTRODUCED_IN_1.20.5",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": {
                        "executor": "generator_handoff_data_component",
                        "canonical_leaf": leaf,
                        "validator_profile": "java_syntax",
                    },
                }
        elif leaf in {"minecraft/dimension/registry", "minecraft/biome/registry"}:
            if v < Version("1.16"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "CUSTOM_BIOMES_AND_DIMENSIONS_INTRODUCED_IN_1.16",
                }
            else:
                suffix = leaf.split("/")[1]
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": {
                        "executor": f"generator_handoff_{suffix}",
                        "canonical_leaf": leaf,
                        "validator_profile": "json_schema",
                    },
                }
        elif leaf in {"minecraft/worldgen/configured_feature", "minecraft/advancement/requirement"}:
            suffix = leaf.split("/")[1]
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": {
                    "executor": f"generator_handoff_{suffix}",
                    "canonical_leaf": leaf,
                    "validator_profile": "json_schema",
                },
            }
        elif leaf == "minecraft/item/component":
            if v < Version("1.20.5"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_DATA_COMPONENTS_REQUIRE_MC_1_20_5",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "not_reviewed",
                    "reason": "DATA_COMPONENT_LEAF_UNDER_REVIEW",
                }
        elif leaf.startswith("minecraft/recipe/") or leaf.startswith("minecraft/tag/"):
            if not is_modern_recipes:
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "not_reviewed",
                    "reason": "LEAF_NOT_YET_REVIEWED",
                }
        else:
            leaf_bindings[leaf] = {
                "state": "not_reviewed",
                "reason": "LEAF_NOT_YET_REVIEWED",
            }

    facts = dict(base_facts)
    facts["capabilities"] = {
        "REGISTER_ITEM": True,
        "REGISTER_BLOCK": True,
        "RECIPE_CRAFTING": is_modern_recipes,
        "RECIPE_SMELTING": is_modern_recipes,
        "LOOT_TABLE": True,
        "TAGS": is_modern_recipes,
        "ITEM_SETTINGS": True,
        "item": True,
        "block": True,
        "recipe": is_modern_recipes,
        "loot": True,
        "tag": is_modern_recipes,
        "ENTITY_EXISTS": True,
        "GUI_EXISTS": True,
        "NETWORK_PACKET": True,
        "BLOCK_ENTITY_EXISTS": True,
        "DATA_COMPONENT": v >= Version("1.20.5"),
        "WORLDGEN_FEATURE": True,
        "DIMENSION": v >= Version("1.16"),
        "BIOME": v >= Version("1.16"),
        "STATUS_EFFECT": True,
        "SOUND_EVENT": True,
        "PARTICLE_TYPE": True,
        "ADVANCEMENT": True,
        "CUSTOM_ITEM_BEHAVIOR": True,
        "CUSTOM_BLOCK_BEHAVIOR": True,
        "UNSUPPORTED": False,
    }
    facts["api_symbols"] = {
        "register_item": "Registry.register",
        "register_block": "Registry.register",
        "resource_key_create": "ResourceKey.create",
        "builtin_item_registry": (
            "BuiltInRegistries.ITEM" if is_modern_registry else "Registry.ITEM"
        ),
        "builtin_block_registry": (
            "BuiltInRegistries.BLOCK" if is_modern_registry else "Registry.BLOCK"
        ),
        "identifier_factory": (
            "Identifier.of" if is_modern_id else "new Identifier"
        ),
        "item_stacks_to": ".stacksTo",
        "registries_item": (
            "Registries.ITEM" if is_modern_registry else "Registry.ITEM_KEY"
        ),
        "registries_block": (
            "Registries.BLOCK" if is_modern_registry else "Registry.BLOCK_KEY"
        ),
        "EntityType": "net.minecraft.entity.EntityType",
        "BlockEntityType": "net.minecraft.block.entity.BlockEntityType",
        "StatusEffect": "net.minecraft.entity.effect.StatusEffect",
        "SoundEvent": "net.minecraft.sound.SoundEvent",
        "ParticleType": "net.minecraft.particle.ParticleType",
        "ScreenHandlerType": "net.minecraft.screen.ScreenHandlerType",
    }
    if v >= Version("1.20.5"):
        facts["api_symbols"]["ComponentType"] = "net.minecraft.component.ComponentType"
    facts["schemas"] = {
        k: ARTIFACT_SCHEMAS[k] for k in admitted_templates if k in ARTIFACT_SCHEMAS
    }
    facts["artifact_rules"] = rules
    facts["leaf_bindings"] = leaf_bindings
    facts["replacements"] = {
        "new Identifier": "Identifier.of" if is_modern_id else "new Identifier"
    }
    return facts


def populate_catalog(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[int, str]:
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
        facts = build_version_facts(
            minecraft, base_facts=bundle["host_facts"], hashes=hashes
        )

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

        raw = {
            "target": bundle["target"],
            "host_facts": facts,
            "source": "HOST",
        }
        ctx = ResolvedVersionContext(_encode(raw))
        row["context_id"] = ctx.context_id

        new_bundles.append(ctx.to_dict())
        new_evidence_rows.append(row)

    auto_context_id = new_bundles[0]["context_id"]
    catalog["auto_context_id"] = auto_context_id
    catalog["bundles"] = new_bundles
    evidence_report["versions"] = new_evidence_rows

    catalog_path.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence_path.write_text(
        json.dumps(evidence_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return len(new_bundles), auto_context_id


if __name__ == "__main__":
    count, auto_id = populate_catalog()
    print(f"Populated {count} bundles. AUTO context_id: {auto_id}")
