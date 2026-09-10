"""Requirement-bound content records and fail-closed lowering to atomic facts."""

from collections.abc import Mapping
from hashlib import sha256
import re

from .atomic_slot_executor import SlotFillError
from .complete_spec import AssetRequest, ProductionModule
from .implementation_fact import FactProvenance, FactType, ImplementationFact
from .implementation_template_renderer import render_template
from .task_template_catalog import load_template
from .task_template_runner import run_record_template


def compile_content_graph(
    prompt,
    router,
    *,
    request_catalog=None,
    research=None,
    progress=None,
    checkpoint=None,
):
    from .atomic_design_pipeline import ALL_DESIGN_SLOTS, _sanitize_stem
    from .design_requirement_contract import _active_requirement_ledger

    if router is None:
        raise SlotFillError("CONTENT_GRAPH_UNRESOLVED: no router supplied")
    requirements = (request_catalog or {}).get("requirements")
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
            {**context, "existing_entities": list(entities.values())},
        )
        if not nodes:
            raise SlotFillError(f"CONTENT_REQUIREMENT_UNRESOLVED: {rid} has no content")
        for node in nodes:
            eid = node["entity_id"]
            prior = entities.get(eid)
            if prior and any(prior[key] != node[key] for key in ("kind", "role")):
                raise SlotFillError(f"CONTENT_ENTITY_CONFLICT: {eid}")
            if prior is None:
                entities[eid] = {**node, "requirement_refs": [], "source_clauses": []}
            entities[eid]["requirement_refs"].append(rid)
            entities[eid]["source_clauses"].append(statement)
        owned.append(context)

    for context in owned:
        for edge in records(
            "design/content_relation", {**context, "entity_ids": list(entities)}
        ):
            if edge["source_id"] not in entities or edge["target_id"] not in entities:
                raise SlotFillError(f"CONTENT_RELATION_DANGLING: {edge}")
            key = (edge["relation_type"], edge["source_id"], edge["target_id"])
            if key in relation_keys:
                raise SlotFillError(f"CONTENT_RELATION_DUPLICATE: {key}")
            relation_keys.add(key)
            relations.append({**edge, "parent_requirement": context["requirement_id"]})
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

    facts, modules, assets = [], [], []
    mod_id = _sanitize_stem(prompt) + "_mod"
    capabilities = {}
    properties_by_id = {}
    for eid, node in entities.items():
        context = {
            "entity": node,
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
        capability = records("design/content_capability", context)
        if len(capability) != 1:
            raise SlotFillError(
                f"CONTENT_CAPABILITY_UNRESOLVED: {eid} needs exactly one base capability"
            )
        try:
            fact_type = FactType(capability[0]["fact_type"])
        except ValueError as exc:
            raise SlotFillError(
                f"CONTENT_CAPABILITY_UNSUPPORTED: {eid}: {capability}"
            ) from exc
        if fact_type not in {
            FactType.ITEM_EXISTS,
            FactType.BLOCK_EXISTS,
            FactType.ENTITY_EXISTS,
            FactType.GUI_EXISTS,
            FactType.NETWORK_PACKET,
            FactType.BLOCK_ENTITY_EXISTS,
            FactType.DATA_COMPONENT,
            FactType.WORLDGEN_FEATURE,
            FactType.DIMENSION,
            FactType.BIOME,
            FactType.STATUS_EFFECT,
            FactType.SOUND_EVENT,
            FactType.PARTICLE_TYPE,
            FactType.ENTITY_LOOT,
            FactType.ADVANCEMENT,
            FactType.EQUIPMENT_ARMOR,
            FactType.CUSTOM_ITEM_BEHAVIOR,
            FactType.CUSTOM_BLOCK_BEHAVIOR,
            FactType.CRAFTING_RECIPE,
            FactType.SMELTING_RECIPE,
            FactType.REGISTRY_TAG,
        }:
            raise SlotFillError(
                f"CONTENT_CAPABILITY_UNSUPPORTED: {eid}: {fact_type.value}"
            )
        capabilities[eid] = fact_type
        props = {}
        visual_properties = {
            "shape",
            "material",
            "main_color",
            "accent",
            "surface",
            "silhouette",
            "transparency",
            "tileability",
        }
        allowed_properties = {
            FactType.ITEM_EXISTS: visual_properties | {"display_name", "stack_limit"},
            FactType.BLOCK_EXISTS: visual_properties | {"display_name"},
            FactType.ENTITY_EXISTS: visual_properties
            | {
                "display_name",
                "category",
                "health",
                "attack_damage",
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
            FactType.CRAFTING_RECIPE: {
                "recipe_kind",
                "count",
                "pattern_1",
                "pattern_2",
                "pattern_3",
            },
            FactType.SMELTING_RECIPE: {"cooking_type", "experience", "cookingtime"},
            FactType.REGISTRY_TAG: {"registry_kind"},
        }[fact_type]
        for prop in records(
            "design/content_property",
            {**context, "allowed_properties": sorted(allowed_properties)},
        ):
            key, value = prop["property"], prop["value"]
            if key in props:
                raise SlotFillError(f"CONTENT_PROPERTY_DUPLICATE: {eid}.{key}")
            if key not in allowed_properties:
                raise SlotFillError(f"CONTENT_PROPERTY_UNSUPPORTED: {eid}.{key}")
            props[key] = value
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
                "attack_damage",
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
                "attack_damage",
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
        fact_kind_map = {
            FactType.ITEM_EXISTS: "item",
            FactType.BLOCK_EXISTS: "block",
            FactType.ENTITY_EXISTS: "entity",
            FactType.GUI_EXISTS: "gui",
            FactType.NETWORK_PACKET: "networking",
            FactType.BLOCK_ENTITY_EXISTS: "machine",
            FactType.DATA_COMPONENT: "custom_java",
            FactType.WORLDGEN_FEATURE: "structure",
            FactType.DIMENSION: "dimension",
            FactType.BIOME: "biome",
            FactType.STATUS_EFFECT: "effect",
            FactType.SOUND_EVENT: "custom_java",
            FactType.PARTICLE_TYPE: "custom_java",
            FactType.ENTITY_LOOT: "loot",
            FactType.ADVANCEMENT: "advancement",
            FactType.EQUIPMENT_ARMOR: "armor",
            FactType.CUSTOM_ITEM_BEHAVIOR: "item",
            FactType.CUSTOM_BLOCK_BEHAVIOR: "block",
        }
        kind = fact_kind_map[fact_type]
        config = {
            "name": props["display_name"],
            "requirement_refs": node["requirement_refs"],
            "implementation_obligations": [node["role"]],
            "reason": node["role"],
        }
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
                    "attack_damage": float(props["attack_damage"]),
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
                mold_template_id = "asset/item_sprite"
                asset_kind = "item"
                w, h = 16, 16
            elif kind == "block":
                mold_template_id = "asset/block_tile"
                asset_kind = "block"
                w, h = 16, 16
            elif kind in {"entity", "boss", "npc"}:
                mold_template_id = "asset/entity_texture"
                asset_kind = "entity"
                w, h = 64, 64
            elif kind == "gui":
                mold_template_id = "asset/gui_panel"
                asset_kind = "gui"
                w, h = 256, 256
            else:
                mold_template_id = "asset/item_sprite"
                asset_kind = "item"
                w, h = 16, 16

            mold_tmpl = load_template(mold_template_id)
            prompt_text = render_template(
                mold_tmpl, {"visual_description": visual_desc}
            )
            assets.append(
                AssetRequest(
                    asset_id=f"texture_{asset_kind}_{eid}",
                    kind=asset_kind,
                    target_path=f"assets/{mod_id}/textures/{asset_kind}/{eid}.png",
                    width=w,
                    height=h,
                    prompt=prompt_text,
                )
            )

    module_by_id = {m.module_id: m for m in modules}
    module_deps = {m.module_id: list(m.depends_on) for m in modules}

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
                module_deps[source].append(target)
                module_by_id[source].config.setdefault(rel_type, []).append(target)

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
                module_deps[target].append(source)
                module_by_id[target].config.setdefault("unlocked_by", []).append(source)

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
                module_deps[source].append(target)
                module_by_id[source].config["opens_gui"] = target

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
                module_deps[source].append(target)
                module_by_id[source].config["controls"] = target

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
                module_deps[source].append(target)
                module_by_id[source].config["spawns"] = target

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
                module_deps[source].append(target)
                module_by_id[source].config["transports_to"] = target

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
                module_deps[source].append(target)
                module_by_id[source].config.setdefault("displays", []).append(target)

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
                module_deps[source].append(target)
                module_by_id[source].config["sync_packet"] = target

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
            expected = {
                "item": FactType.ITEM_EXISTS,
                "block": FactType.BLOCK_EXISTS,
            }.get(registry_kind)
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
            kind = "tag"
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
                        rows = [
                            key
                            for key in ("pattern_1", "pattern_2", "pattern_3")
                            if key in props
                        ]
                        if rows != [f"pattern_{i + 1}" for i in range(len(rows))]:
                            raise ValueError("pattern row gap")
                        value["pattern"] = [props[key] for key in rows]
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
            kind = "recipe"
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
        resource_inputs(fact, mod_id)
        facts.append(fact)
        modules.append(
            ProductionModule(
                eid,
                kind,
                {
                    "requirement_refs": node["requirement_refs"],
                    "implementation_obligations": [node["role"]],
                    "reason": node["role"],
                },
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
        "acceptance_tests": [n["role"] for n in entities.values()],
        "_design_slots": by_slot,
        "_design_decisions": decisions,
        "_implementation_facts": facts,
        "_content_domains": sorted({m.kind for m in modules}),
        "_content_entities": list(entities.values()),
        "_content_relations": relations,
        "_research_facts": research_facts,
    }
