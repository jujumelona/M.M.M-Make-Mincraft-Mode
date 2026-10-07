"""Requirement-bound content records and fail-closed lowering to atomic facts."""

from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
import re
import json

from .spec_identity import SPEC_ID_MAX_LENGTH, canonical_spec_id

class SlotFillError(RuntimeError):
    pass

ALL_DESIGN_SLOTS: tuple[str, ...] = (
    "design/audio_identity",
    "design/combat_role",
    "design/content_scale",
    "design/core_action",
    "design/core_loop",
    "design/core_loop_step",
    "design/crafting_role",
    "design/economy_sink",
    "design/economy_source",
    "design/enemy_role",
    "design/exploration_target",
    "design/failure_condition",
    "design/first_goal",
    "design/goal_prerequisite",
    "design/machine_role",
    "design/network_requirement",
    "design/npc_role",
    "design/persistence_requirement",
    "design/player_fantasy",
    "design/progression_condition",
    "design/progression_edge",
    "design/progression_node",
    "design/resource_sink",
    "design/resource_source",
    "design/reward",
    "design/risk",
    "design/texture_requirement",
    "design/theme",
    "design/ui_requirement",
    "design/unlock",
    "design/visual_identity",
    "design/world_interaction",
)


def _sanitize_stem(name: str) -> str:
    cleaned = "".join(c if (c.isascii() and c.isalnum()) else "_" for c in name.lower())
    parts = [p for p in cleaned.split("_") if p]
    stem = "_".join(parts)
    if not stem:
        stem = f"mmm_{sha256(name.encode('utf-8')).hexdigest()[:10]}"
    if not stem[0].isalpha() or not stem[0].isascii():
        stem = f"mod_{stem}"
    return stem[:30]


_CONTENT_ENTITY_ID_MAX_LENGTH = SPEC_ID_MAX_LENGTH


def _collision_safe_entity_id(
    node: Mapping,
    requirement_id: str,
    entities: Mapping[str, Mapping],
) -> str:
    """Keep authored IDs when possible; deterministically split hard kind collisions."""

    entity_id = str(node["entity_id"])
    prior = entities.get(entity_id)
    if prior is None or prior.get("kind") == node.get("kind"):
        return entity_id

    seed = "\x1f".join(
        (
            requirement_id,
            entity_id,
            str(node.get("kind", "")),
            str(node.get("role", "")),
        )
    )
    digest = sha256(seed.encode("utf-8")).hexdigest()
    stem = entity_id.rstrip("_") or "entity"

    # The model only proposes a readable stem. Global uniqueness is a host
    # responsibility: preserve as much of that stem as possible and bind the
    # conflicting semantic record to a deterministic hash suffix.
    for suffix_length in range(10, len(digest) + 1, 2):
        prefix_budget = _CONTENT_ENTITY_ID_MAX_LENGTH - suffix_length - 1
        prefix = stem[:prefix_budget].rstrip("_") or "entity"
        resolved = f"{prefix}_{digest[:suffix_length]}"
        collision = entities.get(resolved)
        if collision is None:
            return resolved
        if collision.get("kind") == node.get("kind"):
            return resolved

    raise SlotFillError(f"CONTENT_ENTITY_ID_EXHAUSTED: {entity_id}")


def _native_resource_module_config(
    fact_type,
    normalized_inputs: Mapping[str, object],
    node: Mapping[str, object],
) -> dict[str, object]:
    config: dict[str, object] = {
        "requirement_refs": list(node["requirement_refs"]),
        "implementation_obligations": list(node["implementation_obligations"]),
        "reason": node["role"],
    }
    if fact_type == FactType.REGISTRY_TAG:
        registry = {
            "item": "items",
            "block": "blocks",
            "entity_type": "entity_types",
        }.get(str(normalized_inputs.get("registry_kind") or ""))
        values = normalized_inputs.get("members")
        if registry is None or not isinstance(values, list) or not values:
            raise SlotFillError("CONTENT_TAG_NATIVE_CONFIG_INVALID")
        config.update({
            "registry": registry,
            "values": list(values),
            "replace": False,
        })
    return config


def _resource_definition_has_targets(kind: str, entities: Mapping[str, Mapping]) -> bool:
    if kind in {"crafting_recipe", "smelting_recipe"}:
        return any(node.get("kind") == "item" for node in entities.values())
    if kind == "registry_tag":
        return any(
            node.get("kind") in {"item", "block", "entity"}
            for node in entities.values()
        )
    return False


def _materialize_authored_resource_definitions(
    entities: dict[str, dict],
    host_constraints: Mapping[str, object],
    owned_requirements: list[dict],
) -> tuple[str, ...]:
    """Derive resource-definition nodes only after primary content exists.

    Engineering data-resource rows never become semantic requirements.  When a row
    explicitly names a recipe/tag kind and compatible primary targets exist, the host
    may introduce the resource node and bind its provenance to the already-authored
    concrete content requirements.  No synthetic requirement IDs are invented.
    """

    rows = host_constraints.get("data_resources")
    if not isinstance(rows, list) or not rows:
        return ()

    requirement_refs = [
        str(context["requirement_id"])
        for context in owned_requirements
        if isinstance(context.get("requirement_id"), str)
        and str(context["requirement_id"]).strip()
    ]
    source_clauses = [
        str(context["requirement"])
        for context in owned_requirements
        if isinstance(context.get("requirement"), str)
        and str(context["requirement"]).strip()
    ]
    if not requirement_refs or not source_clauses:
        return ()

    created: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        kind = resource_definition_kind_for_data_resource(row.get("kind", ""))
        if kind is None or not _resource_definition_has_targets(kind, entities):
            continue

        encoded = json.dumps(
            {
                "index": index,
                "record": dict(row),
                "kind": kind,
                "requirement_refs": requirement_refs,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = sha256(encoded.encode("utf-8")).hexdigest()[:12]
        role = str(row.get("purpose") or f"{kind} resource").strip()
        if not role:
            role = f"{kind} resource"
        role = role[:128]
        stem = _sanitize_stem(str(row.get("owner") or role or kind))
        entity_id = canonical_spec_id(f"{stem}_{kind}_{digest}")

        prior = entities.get(entity_id)
        if prior is None:
            entities[entity_id] = {
                "entity_id": entity_id,
                "kind": kind,
                "role": role,
                "requirement_refs": list(requirement_refs),
                "source_clauses": list(source_clauses),
                "implementation_obligations": [role],
                "host_derived_resource_definition": True,
            }
            created.append(entity_id)
            continue

        if prior.get("kind") != kind:
            raise SlotFillError(
                f"CONTENT_RESOURCE_ID_CONFLICT: {entity_id}: "
                f"{prior.get('kind')} != {kind}"
            )
        for requirement_id in requirement_refs:
            if requirement_id not in prior["requirement_refs"]:
                prior["requirement_refs"].append(requirement_id)
        for statement in source_clauses:
            if statement not in prior["source_clauses"]:
                prior["source_clauses"].append(statement)
        if role not in prior["implementation_obligations"]:
            prior["implementation_obligations"].append(role)

    return tuple(created)


from .complete_spec import AssetRequest, ProductionModule
from .content_design_contract import (
    CONTENT_FACT_TO_PRODUCTION_KIND,
    REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE,
    fact_type_for_content_kind,
    resource_definition_kind_for_data_resource,
)
from .implementation_fact import FactProvenance, FactType, ImplementationFact
from .task_template_runner import run_record_template


def compile_content_graph(
    prompt,
    router,
    *,
    request_catalog=None,
    research=None,
    progress=None,
    checkpoint=None,
    mod_id: str | None = None,
):
    from .design_requirement_contract import _active_requirement_ledger

    if router is None:
        raise SlotFillError("CONTENT_GRAPH_UNRESOLVED: no router supplied")
    catalog = request_catalog or {}
    requirements = catalog.get("requirements")
    host_constraints = catalog.get("host_constraints", {})
    if not isinstance(host_constraints, Mapping):
        raise SlotFillError("CONTENT_HOST_CONSTRAINTS_INVALID")
    if requirements is None:
        requirements = list(_active_requirement_ledger(prompt))
    if request_catalog is not None and not requirements:
        raise SlotFillError(
            "CONTENT_REQUIREMENT_UNRESOLVED: explicit catalog has no requirements"
        )
    if not requirements:
        requirements = [
            {
                "requirement_id": "req_" + sha256(prompt.encode()).hexdigest()[:16],
                "statement": prompt,
            }
        ]
    entities, relations, decisions = {}, [], []
    requirement_ids = set()
    relation_keys = set()
    progress = progress if progress is not None else {}

    def save(binding, accepted):
        progress[binding] = accepted
        if checkpoint is not None:
            checkpoint(binding, accepted)

    def records(identifier, context):
        return run_record_template(
            router,
            identifier,
            context=context,
            allowed_refs=(),
            progress=progress,
            checkpoint=save,
        )["records"]

    # First discover nodes for each requirement; then bind edges against the complete host graph.
    owned = []
    content_requirements = []
    research_facts = []
    for req in requirements:
        rid = req.get("requirement_id")
        span = req.get("source_span") or {}
        statement = (
            req.get("statement")
            or req.get("semantic_statement")
            or req.get("authored_text")
            or span.get("text")
        )
        if (
            not isinstance(rid, str)
            or not rid
            or rid in requirement_ids
            or not isinstance(statement, str)
            or not statement.strip()
        ):
            raise SlotFillError(
                "CONTENT_REQUIREMENT_INVALID: unique ID and authored statement required"
            )
        requirement_ids.add(rid)
        context = {"requirement_id": rid, "requirement": statement}
        if isinstance(req.get("design_context"), Mapping):
            context["design_context"] = deepcopy(req["design_context"])
        if "coverage_ref" in req:
            from .authored_content_contract import CONTENT_CONCERN_MINIMUM_ENTITY_COUNT
            from .content_design_contract import CONTENT_CONCERN_KINDS

            ref = req["coverage_ref"]
            concern = str(ref).removeprefix("resources_and_ui.")
            from .authored_content_contract import CONTENT_GRAPH_HOST_CONSTRAINT_CONCERNS

            if concern in CONTENT_GRAPH_HOST_CONSTRAINT_CONCERNS:
                raise SlotFillError(
                    f"CONTENT_HOST_CONSTRAINT_AS_REQUIREMENT: {rid}: {ref}"
                )
            expected_minimum = CONTENT_CONCERN_MINIMUM_ENTITY_COUNT.get(concern)
            if (
                ref != f"resources_and_ui.{concern}"
                or concern not in CONTENT_CONCERN_KINDS
                or expected_minimum is None
                or req.get("allowed_content_kinds") != list(CONTENT_CONCERN_KINDS[concern])
                or req.get("minimum_entity_count") != expected_minimum
                or not req.get("source_records")
            ):
                raise SlotFillError(f"CONTENT_REQUIREMENT_BINDING_INVALID: {rid}")
            context.update({
                "coverage_ref": ref,
                "allowed_content_kinds": list(CONTENT_CONCERN_KINDS[concern]),
                "minimum_entity_count": expected_minimum,
                "source_records": deepcopy(req["source_records"]),
            })
            content_requirements.append({
                "requirement_id": rid,
                "coverage_ref": ref,
                "source_records": deepcopy(req["source_records"]),
            })
        grounded = []
        requested_refs = req.get("evidence_refs", ())
        matched_refs = set()
        for evidence in (research or {}).get("evidence", ()):
            if not isinstance(evidence, Mapping):
                raise SlotFillError("CONTENT_RESEARCH_EVIDENCE_INVALID")
            ref = evidence.get("evidence_id") or evidence.get("source_ref")
            owner = evidence.get("parent_requirement") or evidence.get("requirement_id")
            if owner != rid and ref not in requested_refs:
                continue
            if not isinstance(ref, str) or not ref:
                raise SlotFillError("CONTENT_RESEARCH_SOURCE_REQUIRED")
            matched_refs.add(ref)
            for fact in records(
                "design/research_fact",
                {**context, "source_ref": ref, "evidence_shard": dict(evidence)},
            ):
                row = {
                    **fact,
                    "parent_requirement": rid,
                    "source_ref": ref,
                    "source_locator": evidence.get(
                        "source_locator", evidence.get("url", "")
                    ),
                }
                grounded.append(row)
                research_facts.append(row)
        if set(requested_refs) - matched_refs:
            raise SlotFillError(
                f"CONTENT_RESEARCH_SOURCE_MISSING: {set(requested_refs) - matched_refs}"
            )
        if grounded:
            context["research_facts"] = grounded

        nodes = records(
            "design/content_entity",
            {**context, "existing_entities": [
                {key: node[key] for key in ("entity_id", "kind", "role")}
                for node in entities.values()
            ]},
        )
        if len(nodes) < context.get("minimum_entity_count", 0):
            raise SlotFillError(f"CONTENT_REQUIREMENT_UNIMPLEMENTED: {rid}")
        # Generic behavioral requirements may still have no concrete content.
        # Canonical resource obligations above require a bound implementation.
        for node in nodes:
            eid = _collision_safe_entity_id(node, rid, entities)
            if eid != node["entity_id"]:
                node = {**node, "entity_id": eid}
            prior = entities.get(eid)
            if prior is None:
                entities[eid] = {
                    **node, "requirement_refs": [], "source_clauses": [],
                    "implementation_obligations": [],
                }
            if node["role"] not in entities[eid]["implementation_obligations"]:
                entities[eid]["implementation_obligations"].append(node["role"])
            entities[eid]["requirement_refs"].append(rid)
            entities[eid]["source_clauses"].append(statement)
        owned.append(context)

    _materialize_authored_resource_definitions(
        entities,
        host_constraints,
        owned,
    )

    # Discover each ordered entity pair once, with all of its authored context.
    # Repeating the full pair graph for every concern multiplied model work and
    # made shared relations look like duplicate declarations.
    shared_contexts = []
    for context in owned:
        shared = context.get("design_context")
        if shared is not None and shared not in shared_contexts:
            shared_contexts.append(shared)
    for edge in records("design/content_relation", {
        "requirement": prompt,
        "requirements": [
            {key: context[key] for key in ("requirement_id", "requirement")}
            for context in owned
        ],
        "design_contexts": shared_contexts,
        "entity_ids": list(entities),
        "entities": [
            {key: node[key] for key in ("entity_id", "kind", "implementation_obligations")}
            for node in entities.values()
        ],
    }):
        if edge["source_id"] not in entities or edge["target_id"] not in entities:
            raise SlotFillError(f"CONTENT_RELATION_DANGLING: {edge}")
        key = (edge["relation_type"], edge["source_id"], edge["target_id"])
        if key in relation_keys:
            raise SlotFillError(f"CONTENT_RELATION_DUPLICATE: {key}")
        relation_keys.add(key)
        parents = list(dict.fromkeys(
            entities[edge["source_id"]]["requirement_refs"]
            + entities[edge["target_id"]]["requirement_refs"]
        ))
        relations.append({
            **edge, "parent_requirement": parents[0], "requirement_refs": parents,
        })

    for context in owned:
        allowed = [name.split("/")[-1] for name in ALL_DESIGN_SLOTS]
        decision_slots = set()
        for decision in records(
            "design/decision", {**context, "allowed_slots": allowed}
        ):
            if decision["slot_id"] not in allowed:
                raise SlotFillError(f"CONTENT_DECISION_UNKNOWN: {decision['slot_id']}")
            if decision["slot_id"] in decision_slots:
                raise SlotFillError(
                    f"CONTENT_DECISION_CONFLICT: {context['requirement_id']}.{decision['slot_id']}"
                )
            decision_slots.add(decision["slot_id"])
            decisions.append(
                {**decision, "parent_requirement": context["requirement_id"]}
            )

    entity_contexts = {}
    for eid, node in entities.items():
        related = [
            context for context in owned
            if context["requirement_id"] in node["requirement_refs"]
        ]
        design_contexts = []
        for context in related:
            design_context = context.get("design_context")
            if design_context is not None and design_context not in design_contexts:
                design_contexts.append(design_context)
        entity_contexts[eid] = {
            "entity": {
                key: node[key]
                for key in ("entity_id", "kind", "role", "implementation_obligations")
            },
            "requirements": [
                {key: context[key] for key in ("requirement_id", "requirement")}
                for context in related
            ],
            "design_contexts": design_contexts,
            "relations": [
                edge
                for edge in relations
                if eid in (edge["source_id"], edge["target_id"])
            ],
            "design_decisions": [
                decision
                for decision in decisions
                if decision["parent_requirement"] in node["requirement_refs"]
            ],
            "research_facts": [
                fact
                for fact in research_facts
                if fact["parent_requirement"] in node["requirement_refs"]
            ],
        }

    # Entity kind is already schema-bound to the authoritative host vocabulary.
    # Do not spend a model call asking it to repeat a deterministic kind->FactType map.
    capabilities = {}
    for eid, node in entities.items():
        try:
            capabilities[eid] = fact_type_for_content_kind(node["kind"])
        except ValueError as exc:
            raise SlotFillError(
                f"CONTENT_CAPABILITY_UNSUPPORTED: {eid}: {node.get('kind')!r}"
            ) from exc

    facts, modules, assets = [], [], []
    effective_mod_id = str(mod_id or "").strip() or (_sanitize_stem(prompt) + "_mod")
    properties_by_id = {}
    for eid, node in entities.items():
        context = entity_contexts[eid]
        fact_type = capabilities[eid]
        props = {}
        visual_properties = {
            "shape",
            "material",
            "main_color",
            "accent",
            "surface",
            "silhouette",
        }
        allowed_properties = {
            FactType.ITEM_EXISTS: visual_properties | {"display_name", "stack_limit"},
            FactType.BLOCK_EXISTS: visual_properties | {"display_name"},
            FactType.ENTITY_EXISTS: visual_properties
            | {
                "display_name",
                "category",
                "health",
                "speed",
                "tracking_range",
                "width",
                "height",
                "archetype",
                "behavior",
            },
            FactType.GUI_EXISTS: visual_properties
            | {"display_name", "screen_type", "slot_count"},
            FactType.NETWORK_PACKET: {"display_name", "packet_name", "channel", "direction"},
            FactType.BLOCK_ENTITY_EXISTS: {
                "display_name",
                "sync_type",
                "container_size",
            },
            FactType.DATA_COMPONENT: {
                "display_name",
                "component_name",
                "value_type",
                "codec",
            },
            FactType.WORLDGEN_FEATURE: {
                "display_name",
                "feature_type",
                "step",
                "biomes",
            },
            FactType.DIMENSION: {
                "display_name",
                "dimension_type",
                "ambient_light",
                "coordinate_scale",
            },
            FactType.BIOME: {"display_name", "temperature", "downfall", "precipitation"},
            FactType.STATUS_EFFECT: {"display_name", "category", "color", "beneficial"},
            FactType.SOUND_EVENT: {"display_name", "sound_id", "subtitle", "category"},
            FactType.PARTICLE_TYPE: {
                "display_name",
                "particle_name",
                "override_limiter",
            },
            FactType.ENTITY_LOOT: {"display_name", "loot_table_id", "type"},
            FactType.ADVANCEMENT: {"display_name", "parent", "frame_type"},
            FactType.EQUIPMENT_ARMOR: visual_properties
            | {"display_name", "slot", "defense", "toughness"},
            FactType.CUSTOM_ITEM_BEHAVIOR: {"display_name", "action", "cooldown"},
            FactType.CUSTOM_BLOCK_BEHAVIOR: {"display_name", "trigger", "interaction"},
            FactType.CRAFTING_RECIPE: {"count"},
            FactType.SMELTING_RECIPE: {"cooking_type", "experience", "cookingtime"},
            FactType.REGISTRY_TAG: set(),
        }[fact_type]
        required_properties = {
            FactType.ITEM_EXISTS: {"display_name", "main_color", "shape"},
            FactType.BLOCK_EXISTS: {"display_name", "main_color"},
            FactType.ENTITY_EXISTS: {
                "display_name", "category", "health", "speed", "tracking_range",
                "width", "height", "archetype", "behavior", "main_color",
            },
            FactType.GUI_EXISTS: {"display_name", "screen_type", "main_color"},
            FactType.NETWORK_PACKET: {"display_name", "packet_name", "channel", "direction"},
            FactType.BLOCK_ENTITY_EXISTS: {"display_name", "sync_type", "container_size"},
            FactType.DATA_COMPONENT: {"display_name", "component_name", "value_type", "codec"},
            FactType.WORLDGEN_FEATURE: {"display_name", "feature_type", "step", "biomes"},
            FactType.DIMENSION: {"display_name", "dimension_type", "ambient_light", "coordinate_scale"},
            FactType.BIOME: {"display_name", "temperature", "downfall", "precipitation"},
            FactType.STATUS_EFFECT: {"display_name", "category", "color", "beneficial"},
            FactType.SOUND_EVENT: {"display_name", "sound_id", "category"},
            FactType.PARTICLE_TYPE: {"display_name", "particle_name", "override_limiter"},
            FactType.ENTITY_LOOT: {"display_name", "loot_table_id", "type"},
            FactType.ADVANCEMENT: {"display_name", "frame_type"},
            FactType.EQUIPMENT_ARMOR: {"display_name", "slot", "defense", "toughness", "main_color"},
            FactType.CUSTOM_ITEM_BEHAVIOR: {"display_name", "action", "cooldown"},
            FactType.CUSTOM_BLOCK_BEHAVIOR: {"display_name", "trigger", "interaction"},
            FactType.CRAFTING_RECIPE: {"count"},
            FactType.SMELTING_RECIPE: {"cooking_type", "experience", "cookingtime"},
            FactType.REGISTRY_TAG: set(),
        }[fact_type]
        for prop in records(
            "design/content_property",
            {
                **context,
                "allowed_properties": sorted(allowed_properties),
                "required_properties": sorted(required_properties),
            },
        ):
            key, value = prop["property"], prop["value"]
            if key in props:
                raise SlotFillError(f"CONTENT_PROPERTY_DUPLICATE: {eid}.{key}")
            if key not in allowed_properties:
                raise SlotFillError(f"CONTENT_PROPERTY_UNSUPPORTED: {eid}.{key}")
            props[key] = value

        if fact_type == FactType.CRAFTING_RECIPE:
            recipe_edges = [
                edge
                for edge in relations
                if edge["source_id"] == eid
            ]
            key_edges = sorted(
                (
                    edge
                    for edge in recipe_edges
                    if edge["relation_type"].startswith("key_")
                ),
                key=lambda edge: edge["relation_type"],
            )
            if key_edges:
                props["recipe_kind"] = "shaped"
                symbols = [edge["relation_type"][-1] for edge in key_edges]
                packed = "".join(symbols)
                props["pattern"] = [
                    packed[index:index + 3].ljust(3)
                    for index in range(0, len(packed), 3)
                ]
            else:
                props["recipe_kind"] = "shapeless"

        if fact_type == FactType.REGISTRY_TAG:
            member_edges = [
                edge
                for edge in relations
                if edge["source_id"] == eid and edge["relation_type"] == "contains"
            ]
            member_types = {
                capabilities.get(edge["target_id"])
                for edge in member_edges
            }
            inverse_registry_kinds = {
                target_type: registry_kind
                for registry_kind, target_type
                in REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE.items()
            }
            if (
                not member_edges
                or len(member_types) != 1
                or next(iter(member_types)) not in inverse_registry_kinds
            ):
                raise SlotFillError(f"CONTENT_TAG_MEMBERS_INVALID: {eid}")
            props["registry_kind"] = inverse_registry_kinds[next(iter(member_types))]

        if fact_type == FactType.ENTITY_EXISTS:
            behavior_value = str(props.get("behavior", "")).strip().lower()
            if behavior_value in {"hostile_melee", "neutral_melee"} and "attack_damage" not in props:
                combat_rows = records(
                    "design/content_property",
                    {
                        **context,
                        "allowed_properties": ["attack_damage"],
                        "required_properties": ["attack_damage"],
                    },
                )
                if len(combat_rows) != 1 or combat_rows[0]["property"] != "attack_damage":
                    raise SlotFillError(f"CONTENT_PROPERTY_UNRESOLVED: {eid}.attack_damage")
                props["attack_damage"] = combat_rows[0]["value"]

        if fact_type in {
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
            properties_by_id[eid] = props
            continue
        if "display_name" not in props:
            raise SlotFillError(f"CONTENT_PROPERTY_UNRESOLVED: {eid}.display_name")
        if fact_type == FactType.ENTITY_EXISTS:
            required_entity_properties = {
                "category",
                "health",
                "speed",
                "tracking_range",
                "width",
                "height",
                "archetype",
                "behavior",
                "main_color",
            }
            missing_entity_properties = sorted(required_entity_properties - set(props))
            if missing_entity_properties:
                raise SlotFillError(
                    f"CONTENT_PROPERTY_UNRESOLVED: {eid} missing {missing_entity_properties}"
                )
            for numeric_key in (
                "health",
                "speed",
                "tracking_range",
                "width",
                "height",
            ):
                try:
                    numeric_value = float(props[numeric_key])
                except (TypeError, ValueError) as exc:
                    raise SlotFillError(
                        f"CONTENT_PROPERTY_INVALID: {eid}.{numeric_key}"
                    ) from exc
                if numeric_value <= 0:
                    raise SlotFillError(
                        f"CONTENT_PROPERTY_INVALID: {eid}.{numeric_key}"
                    )
            if "attack_damage" in props:
                try:
                    attack_damage = float(props["attack_damage"])
                except (TypeError, ValueError) as exc:
                    raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.attack_damage") from exc
                if attack_damage < 0 or (
                    props["behavior"].strip().lower() in {"hostile_melee", "neutral_melee"}
                    and attack_damage <= 0
                ):
                    raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.attack_damage")
            elif props["behavior"].strip().lower() in {"hostile_melee", "neutral_melee"}:
                raise SlotFillError(f"CONTENT_PROPERTY_UNRESOLVED: {eid}.attack_damage")

            if props["category"].strip().lower() not in {
                "monster",
                "creature",
                "ambient",
                "water_creature",
                "misc",
            }:
                raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.category")
            if props["archetype"].strip().lower() not in {
                "biped",
                "quadruped",
                "flying",
                "serpentine",
                "construct",
            }:
                raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.archetype")
            if props["behavior"].strip().lower() not in {
                "hostile_melee",
                "neutral_melee",
                "passive",
                "npc",
            }:
                raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.behavior")
            if not re.fullmatch(r"#[0-9A-Fa-f]{6}", props["main_color"].strip()):
                raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.main_color")
        properties_by_id[eid] = props
        facts.append(
            ImplementationFact(
                fact_id=f"{eid}.exists",
                fact_type=fact_type,
                subject=eid,
                provenance=FactProvenance.DESIGN,
                parent_requirement=node["requirement_refs"][0],
                source_clause=node["source_clauses"][0],
                display_name=props["display_name"],
            )
        )
        kind = CONTENT_FACT_TO_PRODUCTION_KIND[fact_type]
        config = {
            "name": props["display_name"],
            "requirement_refs": node["requirement_refs"],
            "implementation_obligations": list(node["implementation_obligations"]),
            "reason": node["role"],
        }
        if kind == "integration":
            config["integration_type"] = "artifact_graph_owned"
        for prop_key, prop_val in props.items():
            if prop_key not in visual_properties and prop_key not in {
                "display_name",
                "stack_limit",
            }:
                config[prop_key] = prop_val
        if fact_type == FactType.ENTITY_EXISTS:
            config.update(
                {
                    "max_health": float(props["health"]),
                    "attack_damage": float(props.get("attack_damage", "0")),
                    "movement_speed": float(props["speed"]),
                    "follow_range": float(props["tracking_range"]),
                    "entity_width": float(props["width"]),
                    "entity_height": float(props["height"]),
                    "archetype": props["archetype"].strip().lower(),
                    "behavior": props["behavior"].strip().lower(),
                    "spawn_group": props["category"].strip().lower(),
                    "main_color": props["main_color"].strip(),
                }
            )
        if "stack_limit" in props:
            raw = props["stack_limit"]
            if kind != "item" or not raw.isdecimal() or not 1 <= int(raw) <= 64:
                raise SlotFillError(f"CONTENT_PROPERTY_INVALID: {eid}.stack_limit")
            config["stack_limit"] = int(raw)
            facts.append(
                ImplementationFact(
                    fact_id=f"{eid}.stack",
                    fact_type=FactType.ITEM_STACK_LIMIT,
                    subject=eid,
                    value=int(raw),
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=node["requirement_refs"][0],
                    source_clause=node["source_clauses"][0],
                )
            )
        modules.append(ProductionModule(eid, kind, config))
        visual = {k: v for k, v in props.items() if k in visual_properties}
        if visual:
            visual_desc = ", ".join(f"{k}: {v}" for k, v in visual.items())
            if kind in {"item", "armor"}:
                asset_kind = "item"
                render_kind = "item.generated"
            elif kind == "block":
                asset_kind = "block"
                render_kind = "block.cube_all"
            elif kind in {"entity", "boss", "npc"}:
                asset_kind = "entity"
                render_kind = "entity.fixed_uv"
            elif kind == "gui":
                asset_kind = "gui"
                render_kind = "gui.sprite"
            else:
                raise ValueError(
                    f"ASSET_KIND_UNSUPPORTED: no texture contract for module kind {kind!r} ({eid})"
                )

            assets.append(
                AssetRequest(
                    asset_id=canonical_spec_id(f"texture_{asset_kind}_{eid}"),
                    kind=asset_kind,
                    visual_description=visual_desc,
                    render_kind=render_kind,
                    subject_id=eid,
                    owner_module_id=eid,
                    container="mod",
                    visual_spec={
                        "role": str(props.get("display_name") or eid),
                        "silhouette": str(visual.get("silhouette") or visual.get("shape") or ""),
                        "materials": [visual["material"]] if visual.get("material") else [],
                        "motifs": [visual["surface"]] if visual.get("surface") else [],
                        "palette": {key: visual[source] for key, source in (("primary", "main_color"), ("accent", "accent")) if visual.get(source)},
                    },
                )
            )

    module_by_id = {m.module_id: m for m in modules}
    module_deps = {m.module_id: list(m.depends_on) for m in modules}

    def require_relation_codegen(module_id, relation_type, target_id):
        module = module_by_id.get(module_id)
        if module is None:
            return
        module.config["requires_custom_generation"] = True
        binding = {"relation": relation_type, "target": target_id}
        bindings = module.config.setdefault("executable_relations", [])
        if binding not in bindings:
            bindings.append(binding)

    for edge in relations:
        source, target = edge["source_id"], edge["target_id"]
        rel_type = edge["relation_type"]
        src_cap = capabilities.get(source)
        tgt_cap = capabilities.get(target)
        if src_cap in {
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
            continue

        if rel_type == "drops":
            valid_drop = (
                src_cap == FactType.BLOCK_EXISTS and tgt_cap == FactType.ITEM_EXISTS
            ) or (
                src_cap == FactType.ENTITY_EXISTS
                and tgt_cap in {FactType.ITEM_EXISTS, FactType.ENTITY_LOOT}
            )
            if not valid_drop:
                raise SlotFillError(f"CONTENT_RELATION_UNSUPPORTED: {edge}")
            if any(
                f.fact_type == FactType.BLOCK_DROP and f.subject == source for f in facts
            ):
                raise SlotFillError(f"CONTENT_DROP_CONFLICT: {source}")
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.drop",
                    fact_type=FactType.BLOCK_DROP,
                    subject=source,
                    object=target,
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if source in module_by_id:
                module_by_id[source].config["drop"] = target
                module_deps[source].append(target)

        elif rel_type in {"requires", "upgrades"}:
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.{rel_type}.{target}",
                    fact_type=FactType.CONTENT_RELATION,
                    subject=source,
                    object=target,
                    value={"relation": rel_type},
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if source in module_by_id:
                if rel_type == "requires":
                    module_deps[source].append(target)
                module_by_id[source].config.setdefault(rel_type, []).append(target)
                require_relation_codegen(source, rel_type, target)

        elif rel_type == "unlocks":
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.unlocks.{target}",
                    fact_type=FactType.CONTENT_RELATION,
                    subject=source,
                    object=target,
                    value={"relation": "unlocks"},
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if target in module_by_id:
                module_by_id[target].config.setdefault("unlocked_by", []).append(source)
                require_relation_codegen(target, "unlocked_by", source)

        elif rel_type == "opens":
            if tgt_cap != FactType.GUI_EXISTS:
                raise SlotFillError(f"CONTENT_RELATION_UNSUPPORTED: {edge}")
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.opens.{target}",
                    fact_type=FactType.CONTENT_RELATION,
                    subject=source,
                    object=target,
                    value={"relation": "opens"},
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if source in module_by_id:
                module_by_id[source].config["opens_gui"] = target
                require_relation_codegen(source, "opens", target)

        elif rel_type == "controls":
            if tgt_cap not in {FactType.BLOCK_ENTITY_EXISTS, FactType.ENTITY_EXISTS}:
                raise SlotFillError(f"CONTENT_RELATION_UNSUPPORTED: {edge}")
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.controls.{target}",
                    fact_type=FactType.CONTENT_RELATION,
                    subject=source,
                    object=target,
                    value={"relation": "controls"},
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if source in module_by_id:
                module_by_id[source].config["controls"] = target
                require_relation_codegen(source, "controls", target)

        elif rel_type == "spawns":
            if tgt_cap != FactType.ENTITY_EXISTS:
                raise SlotFillError(f"CONTENT_RELATION_UNSUPPORTED: {edge}")
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.spawns.{target}",
                    fact_type=FactType.CONTENT_RELATION,
                    subject=source,
                    object=target,
                    value={"relation": "spawns"},
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if source in module_by_id:
                module_by_id[source].config["spawns"] = target
                require_relation_codegen(source, "spawns", target)

        elif rel_type == "transports_to":
            if tgt_cap not in {FactType.DIMENSION, FactType.BIOME}:
                raise SlotFillError(f"CONTENT_RELATION_UNSUPPORTED: {edge}")
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.transports_to.{target}",
                    fact_type=FactType.CONTENT_RELATION,
                    subject=source,
                    object=target,
                    value={"relation": "transports_to"},
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if source in module_by_id:
                module_by_id[source].config["transports_to"] = target
                require_relation_codegen(source, "transports_to", target)

        elif rel_type == "displays":
            if src_cap != FactType.GUI_EXISTS:
                raise SlotFillError(f"CONTENT_RELATION_UNSUPPORTED: {edge}")
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.displays.{target}",
                    fact_type=FactType.CONTENT_RELATION,
                    subject=source,
                    object=target,
                    value={"relation": "displays"},
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if source in module_by_id:
                module_by_id[source].config.setdefault("displays", []).append(target)
                require_relation_codegen(source, "displays", target)

        elif rel_type == "synchronizes":
            if tgt_cap != FactType.NETWORK_PACKET:
                raise SlotFillError(f"CONTENT_RELATION_UNSUPPORTED: {edge}")
            facts.append(
                ImplementationFact(
                    fact_id=f"{source}.synchronizes.{target}",
                    fact_type=FactType.CONTENT_RELATION,
                    subject=source,
                    object=target,
                    value={"relation": "synchronizes"},
                    provenance=FactProvenance.DESIGN,
                    parent_requirement=edge["parent_requirement"],
                )
            )
            if source in module_by_id:
                module_by_id[source].config["sync_packet"] = target
                require_relation_codegen(source, "synchronizes", target)

        else:
            raise SlotFillError(f"CONTENT_RELATION_UNSUPPORTED: {edge}")

    updated_modules = []
    for m in modules:
        deps = tuple(dict.fromkeys(module_deps.get(m.module_id, m.depends_on)))
        if deps != m.depends_on:
            updated_modules.append(
                ProductionModule(
                    m.module_id,
                    m.kind,
                    dict(m.config),
                    depends_on=deps,
                    required_gates=m.required_gates,
                )
            )
        else:
            updated_modules.append(m)
    modules = updated_modules

    for eid, fact_type in capabilities.items():
        if fact_type not in {
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
            continue
        props = properties_by_id[eid]
        edges = [edge for edge in relations if edge["source_id"] == eid]
        node = entities[eid]
        if fact_type == FactType.REGISTRY_TAG:
            registry_kind = props.get("registry_kind")
            expected = REGISTRY_TAG_KIND_TO_TARGET_FACT_TYPE.get(registry_kind)
            if expected is None or any(
                edge["relation_type"] != "contains"
                or capabilities[edge["target_id"]] != expected
                for edge in edges
            ):
                raise SlotFillError(f"CONTENT_TAG_MEMBERS_INVALID: {eid}")
            value = {
                "registry_kind": registry_kind,
                "members": [edge["target_id"] for edge in edges],
            }
            kind = CONTENT_FACT_TO_PRODUCTION_KIND[fact_type]
        else:
            if any(
                capabilities[edge["target_id"]] != FactType.ITEM_EXISTS
                for edge in edges
            ):
                raise SlotFillError(f"CONTENT_RECIPE_ITEM_REQUIRED: {eid}")
            outputs = [
                edge["target_id"]
                for edge in edges
                if edge["relation_type"] == "produces"
            ]
            if len(outputs) != 1:
                raise SlotFillError(f"CONTENT_RECIPE_RESULT_REQUIRED: {eid}")
            inputs = [
                edge["target_id"]
                for edge in edges
                if edge["relation_type"] == "consumes"
            ]
            try:
                if fact_type == FactType.SMELTING_RECIPE:
                    if len(inputs) != 1 or any(
                        edge["relation_type"] not in {"consumes", "produces"}
                        for edge in edges
                    ):
                        raise ValueError("one ingredient required")
                    value = {
                        "cooking_type": props["cooking_type"],
                        "ingredient": inputs[0],
                        "result_id": outputs[0],
                        "experience": float(props["experience"]),
                        "cookingtime": int(props["cookingtime"]),
                    }
                else:
                    mode = props["recipe_kind"]
                    value = {
                        "kind": mode,
                        "result_id": outputs[0],
                        "count": int(props["count"]),
                    }
                    if mode == "shapeless":
                        if any(
                            edge["relation_type"] not in {"consumes", "produces"}
                            for edge in edges
                        ):
                            raise ValueError("unsupported recipe relation")
                        value["ingredients"] = inputs
                    elif mode == "shaped":
                        if any(
                            edge["relation_type"] != "produces"
                            and not (
                                edge["relation_type"].startswith("key_")
                                and len(edge["relation_type"]) == 5
                            )
                            for edge in edges
                        ):
                            raise ValueError("shaped recipe needs key_X relations")
                        pattern = props.get("pattern")
                        if (
                            not isinstance(pattern, list)
                            or not 1 <= len(pattern) <= 3
                            or any(
                                not isinstance(row, str) or len(row) != 3
                                for row in pattern
                            )
                        ):
                            raise ValueError("host-shaped pattern required")
                        value["pattern"] = list(pattern)
                        bindings = [
                            edge
                            for edge in edges
                            if edge["relation_type"].startswith("key_")
                        ]
                        if len({edge["relation_type"] for edge in bindings}) != len(
                            bindings
                        ):
                            raise ValueError("conflicting pattern key")
                        value["key"] = {
                            edge["relation_type"][-1]: edge["target_id"]
                            for edge in bindings
                        }
                    else:
                        raise ValueError("unsupported recipe kind")
            except (KeyError, ValueError) as exc:
                raise SlotFillError(f"CONTENT_RECIPE_UNRESOLVED: {eid}: {exc}") from exc
            kind = CONTENT_FACT_TO_PRODUCTION_KIND[fact_type]
        from .resource_fact_inputs import resource_inputs

        fact = ImplementationFact(
            fact_id=f"{eid}.resource",
            fact_type=fact_type,
            subject=eid,
            value=value,
            provenance=FactProvenance.DESIGN,
            parent_requirement=node["requirement_refs"][0],
            source_clause=node["source_clauses"][0],
        )
        _, normalized_resource_inputs = resource_inputs(fact, effective_mod_id)
        facts.append(fact)
        modules.append(
            ProductionModule(
                eid,
                kind,
                _native_resource_module_config(
                    fact_type,
                    normalized_resource_inputs,
                    node,
                ),
                depends_on=tuple(dict.fromkeys(edge["target_id"] for edge in edges)),
            )
        )

    by_slot = {}
    for decision in decisions:
        by_slot.setdefault(decision["slot_id"], []).append(decision["value"])
    return {
        "title": prompt,
        "pitch": prompt,
        "core_loop": by_slot.get("core_loop", []),
        "progression": [
            d["value"]
            for d in decisions
            if d["slot_id"] in {"first_goal", "progression_condition", "reward"}
        ],
        "combat": {},
        "mod_context": {"prompt": prompt},
        "modules": modules,
        "assets": assets,
        "acceptance_tests": [
            role for node in entities.values()
            for role in node["implementation_obligations"]
        ],
        "_design_slots": by_slot,
        "_design_decisions": decisions,
        "_implementation_facts": facts,
        "_content_domains": sorted({m.kind for m in modules}),
        "_content_entities": list(entities.values()),
        **({"_content_requirements": content_requirements} if content_requirements else {}),
        "_content_relations": relations,
        "_research_facts": research_facts,
        "_mod_id": effective_mod_id,
    }
