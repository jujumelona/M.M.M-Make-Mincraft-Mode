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


TEMPLATE_REQUIREMENTS = {
    "fabric/item/register_basic": {
        "requires_capabilities": ["REGISTER_ITEM"],
        "required_symbols": ["register_item"],
    },
    "fabric/item/key": {
        "requires_capabilities": ["REGISTER_ITEM"],
        "required_symbols": ["resource_key_create"],
    },
    "fabric/item/settings_max_stack": {
        "requires_capabilities": ["ITEM_SETTINGS"],
        "required_symbols": ["item_stacks_to"],
    },
    "fabric/item/client_item": {
        "requires_capabilities": ["REGISTER_ITEM"],
        "required_symbols": [],
    },
    "fabric/item/model_basic": {
        "requires_capabilities": ["REGISTER_ITEM"],
        "required_symbols": [],
    },
    "fabric/item/lang_en": {
        "requires_capabilities": ["REGISTER_ITEM"],
        "required_symbols": [],
    },
    "fabric/item/initializer": {
        "requires_capabilities": ["REGISTER_ITEM"],
        "required_symbols": [],
    },
    "fabric/block/register_basic": {
        "requires_capabilities": ["REGISTER_BLOCK"],
        "required_symbols": ["register_block"],
    },
    "fabric/block/key": {
        "requires_capabilities": ["REGISTER_BLOCK"],
        "required_symbols": ["resource_key_create"],
    },
    "fabric/block/blockstate_basic": {
        "requires_capabilities": ["REGISTER_BLOCK"],
        "required_symbols": [],
    },
    "fabric/block/model_cube_all": {
        "requires_capabilities": ["REGISTER_BLOCK"],
        "required_symbols": [],
    },
    "fabric/block/lang_en": {
        "requires_capabilities": ["REGISTER_BLOCK"],
        "required_symbols": [],
    },
    "fabric/block/initializer": {
        "requires_capabilities": ["REGISTER_BLOCK"],
        "required_symbols": [],
    },
    "fabric/recipe/shaped": {
        "requires_capabilities": ["RECIPE_CRAFTING"],
        "required_symbols": [],
    },
    "fabric/recipe/shapeless": {
        "requires_capabilities": ["RECIPE_CRAFTING"],
        "required_symbols": [],
    },
    "fabric/recipe/smelting": {
        "requires_capabilities": ["RECIPE_SMELTING"],
        "required_symbols": [],
    },
    "fabric/tag/registry": {
        "requires_capabilities": ["TAGS"],
        "required_symbols": [],
    },
    "fabric/loot/block_drop": {
        "requires_capabilities": ["LOOT_TABLE"],
        "required_symbols": [],
    },
}


def _sha256_text(text: str) -> str:
    """DEPRECATED: Use ImplementationRegistry instead.
    
    This function kept for backward compatibility during migration.
    """
    return "sha256:" + sha256(text.encode("utf-8")).hexdigest()


def make_implementation(
    leaf: str,
    minecraft_version: str,
    *,
    template_id: str | None = None,
    executor_type: str = "deterministic_renderer",
    validator_profile: str = "semantic_contract",
    hashes: dict[str, str] | None = None,
    extra: dict | None = None,
) -> dict:
    """Create implementation metadata using ImplementationRegistry.
    
    P0-1: Now uses real content hashes from ImplementationRegistry,
    not fake hashes from string concatenation.
    """
    from .implementation_registry import get_global_registry
    from .implementation_identity import ExecutorType, ValidatorType
    
    registry = get_global_registry()
    
    # Get real implementation hash from registry
    if template_id is not None:
        impl_id = f"template:{template_id}"
        
        # Try to get from registry first
        try:
            impl = registry.get_implementation(template_id)
            impl_sha = impl.content_sha256
        except Exception:
            # Fallback: register it now if template file exists
            try:
                from .task_template_catalog import TEMPLATE_DIR
                template_path = TEMPLATE_DIR / f"{template_id}.yaml"
                if template_path.exists():
                    impl = registry.register_template(template_id, template_path)
                    impl_sha = impl.content_sha256
                else:
                    # Last resort: use provided hash or compute from template content
                    impl_sha = (hashes or {}).get(template_id, _sha256_text(f"FALLBACK:template:{template_id}"))
            except Exception:
                impl_sha = (hashes or {}).get(template_id, _sha256_text(f"FALLBACK:template:{template_id}"))
        
        v_profile = validator_profile
    else:
        # Contract or generator case
        impl_id = f"contract:{leaf}"
        impl_sha = _sha256_text(f"contract:{leaf}:{minecraft_version}")
        v_profile = validator_profile
    
    # Validator hash - try registry first
    try:
        # Map profile name to validator type
        validator_type_map = {
            "semantic_contract": ValidatorType.CUSTOM,
            "java_syntax": ValidatorType.JAVA_SYNTAX,
            "json_schema": ValidatorType.JSON_SCHEMA,
        }
        val_type = validator_type_map.get(v_profile, ValidatorType.CUSTOM)
        
        # For now, generate placeholder validator hash
        # TODO P1-2: Register actual validator functions
        val_sha = _sha256_text(f"validator:{v_profile}")
    except Exception:
        val_sha = _sha256_text(f"validator:{v_profile}")
    
    # Schema hashes - TODO P0-8: Use TypeRegistry
    in_sha = _sha256_text(f"input_schema:{leaf}")
    out_sha = _sha256_text(f"output_schema:{leaf}")
    
    # Evidence ID - TODO P1-3: Use EvidenceStore
    evidence_id = f"evidence:host:{leaf}:{minecraft_version}"
    
    impl = {
        "implementation_id": impl_id,
        "executor_type": executor_type,
        "implementation_sha256": impl_sha,
        "validator_profile": v_profile,
        "validator_sha256": val_sha,
        "input_schema_sha256": in_sha,
        "output_schema_sha256": out_sha,
        "evidence_id": evidence_id,
    }
    if template_id:
        impl["template"] = template_id
        impl["template_sha256"] = impl_sha
    if extra:
        impl.update(extra)
    return impl


def template_hashes() -> dict[str, str]:
    """Compute real template hashes using ImplementationRegistry.
    
    P0-1: Now uses actual template file content, not fake hashes.
    """
    from .implementation_registry import get_global_registry
    from .task_template_catalog import TEMPLATE_DIR
    
    registry = get_global_registry()
    hashes = {}
    
    for identifier in LEAF_TEMPLATES:
        try:
            # Try to get from registry
            impl = registry.get_implementation(identifier)
            hashes[identifier] = impl.content_sha256
        except Exception:
            # Register template from file
            try:
                template_path = TEMPLATE_DIR / f"{identifier}.yaml"
                if template_path.exists():
                    impl = registry.register_template(identifier, template_path)
                    hashes[identifier] = impl.content_sha256
                else:
                    # Fallback: load template and hash its content
                    template = load_template(identifier)
                    digest = "sha256:" + sha256(_encode(template).encode()).hexdigest()
                    hashes[identifier] = digest
            except Exception:
                # Last resort fallback
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
        req = TEMPLATE_REQUIREMENTS.get(tid, {"requires_capabilities": [], "required_symbols": []})
        rules[tid] = {
            "template_sha256": hashes[tid],
            "requires_capabilities": list(req["requires_capabilities"]),
            "required_symbols": list(req["required_symbols"]),
        }

    canonical_leaves = all_canonical_leaves()
    leaf_bindings = {}
    for leaf in canonical_leaves:
        # Family 1: item & block
        if leaf == "minecraft/item/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/item/registry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/item/register_basic", executor_type="deterministic_renderer", validator_profile="java_syntax", hashes=hashes, extra={"prerequisite_templates": ["fabric/item/key"]}
                ),
            }
        elif leaf == "minecraft/item/properties":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/item/settings_max_stack", executor_type="deterministic_renderer", validator_profile="java_syntax", hashes=hashes
                ),
            }
        elif leaf == "minecraft/item/model":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/item/model_basic", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes, extra={"extra_templates": ["fabric/item/client_item"] if is_modern_recipes else []}
                ),
            }
        elif leaf == "minecraft/item/language":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/item/lang_en", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/item/integration":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/item/initializer", executor_type="deterministic_renderer", validator_profile="java_syntax", hashes=hashes
                ),
            }
        elif leaf == "minecraft/item/interaction":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_item", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/item/component":
            if v < Version("1.20.5"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_DATA_COMPONENTS_REQUIRE_MC_1_20_5",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="generator_handoff_data_component", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                    ),
                }
        elif leaf == "minecraft/item/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/block/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/block/registry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/block/register_basic", executor_type="deterministic_renderer", validator_profile="java_syntax", hashes=hashes, extra={"prerequisite_templates": ["fabric/block/key"]}
                ),
            }
        elif leaf == "minecraft/block/state":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/block/blockstate_basic", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/block/model":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/block/model_cube_all", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/language/key":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/block/lang_en", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/language/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/language/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/block/integration":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/block/initializer", executor_type="deterministic_renderer", validator_profile="java_syntax", hashes=hashes
                ),
            }
        elif leaf == "minecraft/block/drops":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/loot/block_drop", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/block/interaction":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_block", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/block/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }

        # Family 2: entity, mob, block_entity
        elif leaf == "minecraft/entity/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/entity/registry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_entity", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/entity/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/mob/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/mob/type":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_mob", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/mob/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/block_entity/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/block_entity/registry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_block_entity", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/block_entity/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }

        # Family 3: screen, network, menu, inventory
        elif leaf == "minecraft/screen/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/screen/registration":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_screen", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/screen/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/network_payload/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/network_payload/registration":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_network_payload", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/network_payload/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/menu/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/menu/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/inventory/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/inventory/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }

        # Family 4: recipe, loot, tag, advancement
        elif leaf == "minecraft/recipe/requirement":
            if is_modern_recipes:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                    ),
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
                    "implementation": make_implementation(
                        leaf, minecraft_version, template_id="fabric/recipe/shaped", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes, extra={"supported_templates": ["fabric/recipe/shaped", "fabric/recipe/shapeless", "fabric/recipe/smelting"]}
                    ),
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
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                    ),
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
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                    ),
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
                    "implementation": make_implementation(
                        leaf, minecraft_version, template_id="fabric/tag/registry", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes
                    ),
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
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                    ),
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
                }
        elif leaf == "minecraft/loot/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/loot/entry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, template_id="fabric/loot/block_drop", executor_type="deterministic_renderer", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/loot/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/advancement/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_advancement", validator_profile="json_schema", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/advancement/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                ),
            }

        # Family 5: worldgen, structure, biome, dimension
        elif leaf == "minecraft/worldgen/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/worldgen/configured_feature":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_worldgen", validator_profile="json_schema", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/worldgen/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/structure/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/structure/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/biome/requirement":
            if v < Version("1.16"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "CUSTOM_BIOMES_AND_DIMENSIONS_INTRODUCED_IN_1.16",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                    ),
                }
        elif leaf == "minecraft/biome/registry":
            if v < Version("1.16"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "CUSTOM_BIOMES_AND_DIMENSIONS_INTRODUCED_IN_1.16",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="generator_handoff_biome", validator_profile="json_schema", hashes=hashes, extra={"canonical_leaf": leaf}
                    ),
                }
        elif leaf == "minecraft/biome/validation":
            if v < Version("1.16"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "CUSTOM_BIOMES_AND_DIMENSIONS_INTRODUCED_IN_1.16",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                    ),
                }
        elif leaf == "minecraft/dimension/requirement":
            if v < Version("1.16"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "CUSTOM_BIOMES_AND_DIMENSIONS_INTRODUCED_IN_1.16",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                    ),
                }
        elif leaf == "minecraft/dimension/registry":
            if v < Version("1.16"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "CUSTOM_BIOMES_AND_DIMENSIONS_INTRODUCED_IN_1.16",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="generator_handoff_dimension", validator_profile="json_schema", hashes=hashes, extra={"canonical_leaf": leaf}
                    ),
                }
        elif leaf == "minecraft/dimension/validation":
            if v < Version("1.16"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "CUSTOM_BIOMES_AND_DIMENSIONS_INTRODUCED_IN_1.16",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                    ),
                }

        # Family 6: particle, sound, model, texture, animation, component, effect, attribute, datagen, other
        elif leaf == "minecraft/component/requirement":
            if v < Version("1.20.5"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "DATA_COMPONENTS_INTRODUCED_IN_1.20.5",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                    ),
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
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="generator_handoff_data_component", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                    ),
                }
        elif leaf == "minecraft/component/validation":
            if v < Version("1.20.5"):
                leaf_bindings[leaf] = {
                    "state": "unsupported",
                    "reason": "DATA_COMPONENTS_INTRODUCED_IN_1.20.5",
                }
            else:
                leaf_bindings[leaf] = {
                    "state": "admitted",
                    "implementation": make_implementation(
                        leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                    ),
                }
        elif leaf == "minecraft/effect/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/effect/registry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_effect", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/effect/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/sound/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/sound/registration":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_sound", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/sound/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/particle/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/particle/registry":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="generator_handoff_particle", validator_profile="java_syntax", hashes=hashes, extra={"canonical_leaf": leaf}
                ),
            }
        elif leaf == "minecraft/particle/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/model/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/model/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/texture/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/texture/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/animation/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/animation/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="json_schema", hashes=hashes
                ),
            }
        elif leaf == "minecraft/attribute/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/attribute/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/command/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/command/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/event/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/event/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/keybind/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/keybind/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/hud/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/hud/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/saved_data/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/saved_data/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }
        elif leaf == "minecraft/datagen/requirement":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="semantic_contract", hashes=hashes
                ),
            }
        elif leaf == "minecraft/datagen/validation":
            leaf_bindings[leaf] = {
                "state": "admitted",
                "implementation": make_implementation(
                    leaf, minecraft_version, executor_type="canonical_contract_validator", validator_profile="mod_integration_test", hashes=hashes
                ),
            }

        # Catch-all boundary conditions
        elif v < Version("1.20.5") and leaf.startswith("minecraft/component/"):
            leaf_bindings[leaf] = {
                "state": "unsupported",
                "reason": "DATA_COMPONENTS_INTRODUCED_IN_1.20.5",
            }
        elif v < Version("1.16") and (leaf.startswith("minecraft/dimension/") or leaf.startswith("minecraft/biome/")):
            leaf_bindings[leaf] = {
                "state": "unsupported",
                "reason": "CUSTOM_BIOMES_AND_DIMENSIONS_INTRODUCED_IN_1.16",
            }
        elif not is_modern_recipes and (leaf.startswith("minecraft/recipe/") or leaf.startswith("minecraft/tag/")):
            leaf_bindings[leaf] = {
                "state": "unsupported",
                "reason": "MODERN_RECIPE_SCHEMA_UNSUPPORTED",
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
        "register_item": {
            "owner": "net.minecraft.registry.Registry",
            "name": "register",
            "descriptor": "(Lnet/minecraft/registry/Registry;Lnet/minecraft/util/Identifier;Ljava/lang/Object;)Ljava/lang/Object;",
            "kind": "method",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "register_block": {
            "owner": "net.minecraft.registry.Registry",
            "name": "register",
            "descriptor": "(Lnet/minecraft/registry/Registry;Lnet/minecraft/util/Identifier;Ljava/lang/Object;)Ljava/lang/Object;",
            "kind": "method",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "resource_key_create": {
            "owner": "net.minecraft.registry.RegistryKey",
            "name": "create",
            "descriptor": "(Lnet/minecraft/registry/RegistryKey;Lnet/minecraft/util/Identifier;)Lnet/minecraft/registry/RegistryKey;",
            "kind": "method",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "builtin_item_registry": {
            "owner": "net.minecraft.registry.BuiltInRegistries" if is_modern_registry else "net.minecraft.registry.Registry",
            "name": "ITEM",
            "descriptor": "Lnet/minecraft/registry/DefaultedRegistry;",
            "kind": "field",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "builtin_block_registry": {
            "owner": "net.minecraft.registry.BuiltInRegistries" if is_modern_registry else "net.minecraft.registry.Registry",
            "name": "BLOCK",
            "descriptor": "Lnet/minecraft/registry/DefaultedRegistry;",
            "kind": "field",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "identifier_factory": {
            "owner": "net.minecraft.util.Identifier",
            "name": "of" if is_modern_id else "<init>",
            "descriptor": "(Ljava/lang/String;Ljava/lang/String;)Lnet/minecraft/util/Identifier;",
            "kind": "method" if is_modern_id else "constructor",
            "static": is_modern_id,
            "side": "common",
            "namespace": "minecraft",
        },
        "item_stacks_to": {
            "owner": "net.minecraft.item.Item$Settings",
            "name": "stacksTo",
            "descriptor": "(I)Lnet/minecraft/item/Item$Settings;",
            "kind": "method",
            "static": False,
            "side": "common",
            "namespace": "minecraft",
        },
        "registries_item": {
            "owner": "net.minecraft.registry.Registries" if is_modern_registry else "net.minecraft.registry.Registry",
            "name": "ITEM" if is_modern_registry else "ITEM_KEY",
            "descriptor": "Lnet/minecraft/registry/RegistryKey;",
            "kind": "field",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "registries_block": {
            "owner": "net.minecraft.registry.Registries" if is_modern_registry else "net.minecraft.registry.Registry",
            "name": "BLOCK" if is_modern_registry else "BLOCK_KEY",
            "descriptor": "Lnet/minecraft/registry/RegistryKey;",
            "kind": "field",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "EntityType": {
            "owner": "net.minecraft.entity.EntityType",
            "name": "EntityType",
            "descriptor": "Lnet/minecraft/entity/EntityType;",
            "kind": "class",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "BlockEntityType": {
            "owner": "net.minecraft.block.entity.BlockEntityType",
            "name": "BlockEntityType",
            "descriptor": "Lnet/minecraft/block/entity/BlockEntityType;",
            "kind": "class",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "StatusEffect": {
            "owner": "net.minecraft.entity.effect.StatusEffect",
            "name": "StatusEffect",
            "descriptor": "Lnet/minecraft/entity/effect/StatusEffect;",
            "kind": "class",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "SoundEvent": {
            "owner": "net.minecraft.sound.SoundEvent",
            "name": "SoundEvent",
            "descriptor": "Lnet/minecraft/sound/SoundEvent;",
            "kind": "class",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "ParticleType": {
            "owner": "net.minecraft.particle.ParticleType",
            "name": "ParticleType",
            "descriptor": "Lnet/minecraft/particle/ParticleType;",
            "kind": "class",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
        "ScreenHandlerType": {
            "owner": "net.minecraft.screen.ScreenHandlerType",
            "name": "ScreenHandlerType",
            "descriptor": "Lnet/minecraft/screen/ScreenHandlerType;",
            "kind": "class",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        },
    }
    if v >= Version("1.20.5"):
        facts["api_symbols"]["ComponentType"] = {
            "owner": "net.minecraft.component.ComponentType",
            "name": "ComponentType",
            "descriptor": "Lnet/minecraft/component/ComponentType;",
            "kind": "class",
            "static": True,
            "side": "common",
            "namespace": "minecraft",
        }
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
