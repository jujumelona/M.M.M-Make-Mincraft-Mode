import json
from copy import deepcopy

import pytest

from minecraft_mod_ai.atomic_design_pipeline import DESIGN_SLOTS, compile_atomic_design
from minecraft_mod_ai.atomic_slot_executor import SlotDefinition, SlotFillError
from minecraft_mod_ai.prompt_fact_types import FactType
from minecraft_mod_ai.task_template_catalog import load_template


class GraphRouter:
    def __init__(self, *, edges=None, capability="ITEM_EXISTS", nodes=None):
        self.calls = []
        self.edges = edges or []
        self.capability = capability
        self.nodes = (
            nodes
            if nodes is not None
            else [
                {
                    "entity_id": "raw_material",
                    "kind": "resource",
                    "role": "Collect raw material",
                },
                {
                    "entity_id": "processed_material",
                    "kind": "processed_material",
                    "role": "Use processed material",
                },
            ]
        )

    def generate_tool_decision(
        self, role, messages, *, tool_name, parameters, **kwargs
    ):
        self.calls.append(tool_name)
        context = json.loads(messages[-1]["content"])
        if tool_name == "submit_one_design_content_entity_count":
            return {"count": len(self.nodes)}
        if tool_name == "submit_one_design_content_relation_count":
            selected = [
                edge
                for edge in self.edges
                if edge["source_id"] == context["source_id"]
                and edge["target_id"] == context["target_id"]
            ]
            return {"count": len(selected)}
        if tool_name == "submit_one_design_continue_record":
            target = str(context.get("target_template") or "")
            return {
                "required": target == "design/research_fact"
                and not context.get("accepted_record_ids")
            }
        single_record = tool_name.startswith("submit_one_")
        normalized_tool = (
            tool_name.replace("submit_one_", "submit_", 1)
            if single_record
            else tool_name
        )
        accepted = context.get("accepted_records", [])
        if normalized_tool == "submit_design_content_entity":
            rows = self.nodes
        elif normalized_tool == "submit_design_content_relation":
            rows = [
                {"relation_type": edge["relation_type"]}
                for edge in self.edges
                if edge["source_id"] == context["source_id"]
                and edge["target_id"] == context["target_id"]
            ]
        elif normalized_tool == "submit_design_content_capability":
            rows = [{"fact_type": self.capability}]
        elif normalized_tool == "submit_design_content_property":
            entity = context.get("entity")
            eid = (
                entity["entity_id"]
                if isinstance(entity, dict)
                else str(context.get("module_id") or "")
            )
            if not eid:
                raise AssertionError(
                    "content-property fake requires entity_id or module_id"
                )
            rows = [
                {
                    "property": "display_name",
                    "value": eid.replace("_", " ").title(),
                },
                {"property": "shape", "value": "faceted chunk"},
            ]
        elif normalized_tool == "submit_design_decision":
            rows = []
        else:
            raise AssertionError(tool_name)
        if single_record:
            if normalized_tool == "submit_design_content_entity":
                index = int(context.get("entity_ordinal", len(accepted) + 1)) - 1
            elif normalized_tool == "submit_design_content_relation":
                index = int(context.get("record_index", len(accepted)))
            elif (
                normalized_tool == "submit_design_content_property"
                and context.get("requested_property")
            ):
                requested = context["requested_property"]
                matching = [row for row in rows if row.get("property") == requested]
                if len(matching) == 1:
                    return deepcopy(matching[0])
                defaults = {
                    "main_color": "#808080",
                    "shape": "faceted chunk",
                    "hardness": "1.5",
                    "attack_damage": "4",
                    "attack_speed": "1.0",
                    "hunger": "4",
                    "saturation": "0.4",
                    "seed_color": "#70A050",
                    "category": "creature",
                    "health": "20",
                    "speed": "0.25",
                    "tracking_range": "32",
                    "width": "0.6",
                    "height": "1.8",
                    "archetype": "biped",
                    "behavior": "passive",
                    "screen_type": "container_9x3",
                    "slot_count": "27",
                    "packet_name": f"{eid}_packet",
                    "channel": f"test:{eid}",
                    "direction": "s2c",
                    "sync_type": "ticking_block_entity",
                    "container_size": "9",
                    "component_name": eid,
                    "value_type": "string",
                    "codec": "Codec.STRING",
                    "feature_type": "ore",
                    "step": "underground_ores",
                    "biomes": "minecraft:plains",
                    "dimension_type": "minecraft:overworld",
                    "ambient_light": "0.0",
                    "coordinate_scale": "1.0",
                    "temperature": "0.8",
                    "downfall": "0.4",
                    "precipitation": "rain",
                    "color": "#808080",
                    "beneficial": "true",
                    "sound_id": f"test:{eid}",
                    "particle_name": eid,
                    "override_limiter": "false",
                    "loot_table_id": f"test:entities/{eid}",
                    "type": "entity",
                    "frame_type": "task",
                    "slot": "chestplate",
                    "defense": "4",
                    "toughness": "1.0",
                    "action": "use",
                    "cooldown": "20",
                    "trigger": "use",
                    "interaction": "activate",
                    "input_item": "minecraft:stone",
                    "output_item": "minecraft:cobblestone",
                    "output_count": "1",
                    "processing_ticks": "20",
                    "max_level": "1",
                    "literal": "test",
                    "message": "test",
                    "permission_level": "0",
                }
                if requested not in defaults:
                    raise AssertionError(f"missing requested property {requested}")
                return {"property": requested, "value": defaults[requested]}
            else:
                index = len(accepted)
            if not 0 <= index < len(rows):
                raise AssertionError(
                    f"single record index out of range for {normalized_tool}: {index}"
                )
            return deepcopy(rows[index])
        if len(accepted) < len(rows):
            return {
                "status": "record",
                "record": deepcopy(rows[len(accepted)]),
                "reason": "",
            }
        return {
            "status": "done" if rows else "not_applicable",
            "record": None,
            "reason": "No additional decision or relation required" if not rows else "",
        }


def test_design_templates_obey_atomicity():
    for identifier in (
        *DESIGN_SLOTS,
        "design/content_entity",
        "design/content_relation",
        "design/content_relation_count",
        "design/content_capability",
        "design/content_property",
        "design/decision",
    ):
        template = load_template(identifier)
        SlotDefinition(identifier, template["record_schema"]).validate_schema()


def test_graph_emits_multiple_entities_instead_of_prompt_stem_module():
    design = compile_atomic_design("Create two materials", GraphRouter())
    assert [m.module_id for m in design["modules"]] == [
        "raw_material",
        "processed_material",
    ]
    assert all(
        f.fact_type == FactType.ITEM_EXISTS for f in design["_implementation_facts"]
    )
    assert design["_content_relations"] == []
    assert design["core_loop"] == []
    assert design["progression"] == []
    assert design["_design_slots"] == {}
    for module in design["modules"]:
        module.validate()
    for asset in design["assets"]:
        asset.validate()
        assert "faceted chunk" in asset.prompt


def test_graph_provenance_is_bound_to_exact_requirement():
    catalog = {
        "requirements": [
            {"requirement_id": "req_material", "statement": "Create two materials"}
        ]
    }
    design = compile_atomic_design(
        "Create two materials", GraphRouter(), request_catalog=catalog
    )
    assert all(
        f.parent_requirement == "req_material" for f in design["_implementation_facts"]
    )
    assert all(f.evidence_refs == () for f in design["_implementation_facts"])


def test_relation_pairs_are_host_scoped_and_cannot_be_dangling():
    design = compile_atomic_design(
        "materials",
        GraphRouter(
            edges=[
                {
                    "relation_type": "drops",
                    "source_id": "missing",
                    "target_id": "raw_material",
                }
            ]
        ),
    )
    assert design["_content_relations"] == []


def test_graph_rejects_unsupported_relation_instead_of_ignoring_it():
    with pytest.raises(SlotFillError, match="RELATION_UNSUPPORTED"):
        compile_atomic_design(
            "materials",
            GraphRouter(
                edges=[
                    {
                        "relation_type": "transports_to",
                        "source_id": "raw_material",
                        "target_id": "processed_material",
                    }
                ]
            ),
        )


@pytest.mark.parametrize("capability", ["UNSUPPORTED"])
def test_unsupported_capability_never_becomes_item(capability):
    with pytest.raises(SlotFillError, match="CAPABILITY_UNSUPPORTED"):
        compile_atomic_design("requested feature", GraphRouter(capability=capability))


def test_no_router_or_empty_prompt_fails_closed():
    with pytest.raises(SlotFillError, match="UNRESOLVED"):
        compile_atomic_design("create material")
    with pytest.raises(ValueError, match="ATOMIC_DESIGN"):
        compile_atomic_design("")


def test_graph_resumes_validated_records_without_recalling_model():
    progress = {}
    first = GraphRouter()
    design = compile_atomic_design("materials", first, progress=progress)
    assert progress and first.calls
    second = GraphRouter()
    resumed = compile_atomic_design("materials", second, progress=progress)
    assert resumed == design
    assert second.calls == []


def test_one_content_record_per_model_call():
    router = GraphRouter()
    compile_atomic_design("materials", router)
    assert router.calls.count("submit_one_design_content_entity") == 2
    assert router.calls.count("submit_one_design_content_entity_count") == 1
    assert "resolve_domains" not in router.calls


def test_graph_passes_real_readiness_without_invented_progression():
    from minecraft_mod_ai.agentic_research_game_design import (
        canonical_game_design,
        validate_ready_design,
    )

    design = compile_atomic_design("materials", GraphRouter())
    ready = validate_ready_design("materials", canonical_game_design(design))
    assert ready["core_loop"] == []
    assert ready["progression"] == []


def test_research_facts_are_bound_to_their_actual_source():
    class ResearchRouter(GraphRouter):
        def generate_tool_decision(
            self, role, messages, *, tool_name, parameters, **kwargs
        ):
            if tool_name in {
                "submit_design_research_fact",
                "submit_one_design_research_fact",
            }:
                context = json.loads(messages[-1]["content"])
                assert context["source_ref"] == "source_b"
                if tool_name == "submit_one_design_research_fact":
                    return {"fact": "Material is brittle"}
                if not context["accepted_records"]:
                    return {
                        "status": "record",
                        "record": {"fact": "Material is brittle"},
                        "reason": "",
                    }
                return {"status": "done", "record": None, "reason": ""}
            return super().generate_tool_decision(
                role, messages, tool_name=tool_name, parameters=parameters, **kwargs
            )

    design = compile_atomic_design(
        "materials",
        ResearchRouter(),
        request_catalog={
            "requirements": [
                {
                    "requirement_id": "req_material",
                    "statement": "materials",
                    "evidence_refs": ["source_b"],
                }
            ]
        },
        research={
            "evidence": [
                {"evidence_id": "source_a", "text": "unrelated"},
                {
                    "evidence_id": "source_b",
                    "text": "Material is brittle",
                    "source_locator": "p2",
                },
            ]
        },
    )
    assert design["_research_facts"] == [
        {
            "fact": "Material is brittle",
            "source_ref": "source_b",
            "source_locator": "p2",
            "parent_requirement": "req_material",
        }
    ]


def test_recipe_content_graph_reaches_artifact_files(tmp_path, monkeypatch):
    from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
    from minecraft_mod_ai.artifact_graph_executor import execute_artifact_graph
    from minecraft_mod_ai.artifact_materializer import ensure_artifact_scaffolding

    class RecipeRouter(GraphRouter):
        def generate_tool_decision(
            self, role, messages, *, tool_name, parameters, **kwargs
        ):
            context = json.loads(messages[-1]["content"])
            if context.get("entity", {}).get("entity_id") == "conversion":
                if tool_name == "submit_one_design_content_capability":
                    return {"fact_type": "CRAFTING_RECIPE"}
                if tool_name == "submit_one_design_content_property":
                    rows = [
                        {"property": "recipe_kind", "value": "shapeless"},
                        {"property": "count", "value": "1"},
                    ]
                    requested = context.get("requested_property")
                    return next(row for row in rows if row["property"] == requested)
                rows = (
                    [{"fact_type": "CRAFTING_RECIPE"}]
                    if tool_name == "submit_design_content_capability"
                    else [
                        {"property": "recipe_kind", "value": "shapeless"},
                        {"property": "count", "value": "1"},
                    ]
                )
                index = len(context.get("accepted_records", []))
                return (
                    {"status": "record", "record": rows[index], "reason": ""}
                    if index < len(rows)
                    else {"status": "done", "record": None, "reason": ""}
                )
            return super().generate_tool_decision(
                role, messages, tool_name=tool_name, parameters=parameters, **kwargs
            )

    router = RecipeRouter(
        edges=[
            {
                "relation_type": "consumes",
                "source_id": "conversion",
                "target_id": "raw_material",
            },
            {
                "relation_type": "produces",
                "source_id": "conversion",
                "target_id": "processed_material",
            },
        ]
    )
    router.nodes.append(
        {"entity_id": "conversion", "kind": "process", "role": "Convert materials"}
    )
    design = compile_atomic_design("materials", router)
    ensure_artifact_scaffolding(
        tmp_path, mod_id="bound_target", package_name="org.demo"
    )
    jobs = expand_facts_to_jobs(
        design["_implementation_facts"],
        mod_id="bound_target",
        package_name="org.demo",
        minecraft_version="1.21.2",
    )
    import minecraft_mod_ai.resolved_version_context as resolved_version_context

    monkeypatch.setattr(
        resolved_version_context,
        "execution_context",
        lambda context, job: None,
    )
    execute_artifact_graph(jobs, base_dir=tmp_path)
    data = json.loads(
        (
            tmp_path / "src/main/resources/data/bound_target/recipe/conversion.json"
        ).read_text()
    )
    assert data["result"]["id"] == "bound_target:processed_material"


def test_normal_planning_projection_keeps_graph_content_and_asset_namespace(
    monkeypatch,
):
    from contextlib import nullcontext

    import minecraft_mod_ai.agentic_research_game_design as host
    import minecraft_mod_ai.planning_authority as authority
    import minecraft_mod_ai.reuse_planner as reuse
    from minecraft_mod_ai.planning_pipeline import PlanningPipeline

    monkeypatch.setattr(
        authority,
        "build_authoritative_request_catalog",
        lambda *a, **k: {
            "requirements": [
                {"requirement_id": "req_material", "statement": "materials"}
            ]
        },
    )
    monkeypatch.setattr(
        authority, "authoritative_request_scope", lambda *a, **k: nullcontext()
    )
    monkeypatch.setattr(reuse, "compile_pre_retrieval_plan", lambda *a, **k: {})
    monkeypatch.setattr(
        host, "research_brief_from_design", lambda *a, **k: {}, raising=False
    )
    pipeline = PlanningPipeline(GraphRouter())
    design, proposal = pipeline._semantic_design(
        "materials for a complex long title with many words",
        planning_state={},
        media_paths=(),
    )
    assert proposal.spec.contents == ()
    assert proposal.spec.boss is None
    assert len(design["_content_entities"]) == 2
    assert all(
        a.target_path.startswith(f"assets/{proposal.spec.mod_id}/")
        for a in design["_atomic_assets"]
    )
