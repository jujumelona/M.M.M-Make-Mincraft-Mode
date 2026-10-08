import pytest

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_production import _compile_new_authored_modules


def test_untyped_authored_plan_is_rejected_before_production() -> None:
    text = (
        "# behavior_contract\nTrade ore.\n"
        "# state_model\nPlayerCredits, Ship, ShipPart, persistence.\n"
        "# verification\nTest transactions."
    )
    plan = AuthoredPlan("space economy", text)

    with pytest.raises(ValueError, match="TYPED_PLAN_REQUIRED"):
        _compile_new_authored_modules(
            plan,
            mod_id="authored_test",
            package_name="example",
            target={
                "minecraft_version": "1.21.1",
                "loader": "fabric",
                "mappings": "none",
            },
        )


def test_not_reviewed_python_generator_is_candidate_content_route() -> None:
    from types import SimpleNamespace

    from minecraft_mod_ai.authored_production import _fact_artifact_route_ready
    from minecraft_mod_ai.implementation_fact import ImplementationFact
    from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
    from minecraft_mod_ai.prompt_fact_types import FactType

    leaf = "minecraft/screen/registration"
    facts = build_version_facts("26.2", base_facts={})
    binding = facts["leaf_bindings"][leaf]
    assert binding["state"] == "not_reviewed"
    assert binding["implementation"]["implementation_id"].startswith("python_generator:")

    resolved = SimpleNamespace(
        leaf_bindings={leaf: binding},
        context_id="candidate-26.2",
    )
    fact = ImplementationFact(
        fact_id="market_ui.exists",
        fact_type=FactType.GUI_EXISTS,
        subject="market_ui",
        source_clause="A server-authoritative market screen is required.",
    )

    assert _fact_artifact_route_ready(fact, resolved)

def test_gui_content_relation_does_not_create_item_initializer_dependency() -> None:
    from types import SimpleNamespace

    from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
    from minecraft_mod_ai.artifact_job import validate_artifact_job_graph
    from minecraft_mod_ai.implementation_fact import ImplementationFact
    from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
    from minecraft_mod_ai.prompt_fact_types import FactType

    host_facts = build_version_facts("26.2", base_facts={})
    resolved = SimpleNamespace(
        minecraft="26.2",
        context_id="candidate-26.2",
        leaf_bindings=host_facts["leaf_bindings"],
    )
    gui = ImplementationFact(
        fact_id="gui_market.exists",
        fact_type=FactType.GUI_EXISTS,
        subject="gui_market",
        source_clause="Market screen exists.",
    )
    relation = ImplementationFact(
        fact_id="gui_market.requires.gui_ship",
        fact_type=FactType.CONTENT_RELATION,
        subject="gui_market",
        object="gui_ship",
        value={"relation": "requires"},
        source_clause="Market screen requires ship screen.",
    )

    jobs = expand_facts_to_jobs(
        (gui, relation),
        mod_id="space_mod",
        package_name="example.space",
        minecraft_version="26.2",
        version_context=resolved,
    )

    assert len(jobs) == 1
    assert jobs[0].canonical_leaf == "minecraft/screen/registration"
    assert jobs[0].template_id == ""
    assert all(job.template_id != "fabric/item/initializer" for job in jobs)
    validate_artifact_job_graph(jobs, module_ids={"gui_market"})



def test_unlock_relations_do_not_form_production_dependency_cycles() -> None:
    from minecraft_mod_ai.authored_production import _normalize_content_build_dependencies
    from minecraft_mod_ai.complete_spec import ProductionModule
    from minecraft_mod_ai.implementation_fact import ImplementationFact
    from minecraft_mod_ai.prompt_fact_types import FactType

    planetary = "gui_planetary"
    library = "gui_library"
    builder = "gui_builder"
    farming = "gui_farming"
    modules = (
        ProductionModule(planetary, "gui", {}, (library, builder, farming)),
        ProductionModule(library, "gui", {}, (builder,)),
        ProductionModule(builder, "gui", {}, (planetary, library)),
        ProductionModule(farming, "gui", {}, (planetary, library)),
    )
    relations = (
        ImplementationFact(
            "p.unlocks.b",
            FactType.CONTENT_RELATION,
            planetary,
            builder,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "p.unlocks.f",
            FactType.CONTENT_RELATION,
            planetary,
            farming,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "l.unlocks.p",
            FactType.CONTENT_RELATION,
            library,
            planetary,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "l.unlocks.b",
            FactType.CONTENT_RELATION,
            library,
            builder,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "l.unlocks.f",
            FactType.CONTENT_RELATION,
            library,
            farming,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "b.unlocks.p",
            FactType.CONTENT_RELATION,
            builder,
            planetary,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "b.unlocks.l",
            FactType.CONTENT_RELATION,
            builder,
            library,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "f.unlocks.p",
            FactType.CONTENT_RELATION,
            farming,
            planetary,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "f.requires.l",
            FactType.CONTENT_RELATION,
            farming,
            library,
            {"relation": "requires"},
        ),
    )

    normalized = _normalize_content_build_dependencies(modules, relations)
    by_id = {module.module_id: module for module in normalized}

    assert by_id[planetary].depends_on == ()
    assert by_id[library].depends_on == ()
    assert by_id[builder].depends_on == ()
    assert by_id[farming].depends_on == (library,)



def test_semantic_relation_cycle_is_removed_but_hard_build_edges_survive() -> None:
    from minecraft_mod_ai.authored_production import _normalize_content_build_dependencies
    from minecraft_mod_ai.complete_spec import ProductionModule
    from minecraft_mod_ai.implementation_fact import ImplementationFact
    from minecraft_mod_ai.prompt_fact_types import FactType

    blueprint = "spacemode_blueprint_types_registry_data_component"
    fabricator = "spacemode_hull_engine_weapon_block_entity_runtime_state_manager"
    registry = "spacemode_ship_module_registry_gui_screen_component_server_side_"
    recipe = "host_crafting_recipe_d458014cd572"

    # This is the dependency shape emitted by the saved proposal that regressed:
    # requires + unlocks + bidirectional displays/upgrades formed one build cycle,
    # while the recipe dependency is a real non-relation production dependency.
    modules = (
        ProductionModule(blueprint, "item", {}, (registry,)),
        ProductionModule(fabricator, "gui", {}, (blueprint, registry)),
        ProductionModule(registry, "gui", {}, (fabricator,)),
        ProductionModule(recipe, "recipe", {}, (blueprint,)),
    )
    relations = (
        ImplementationFact(
            "blueprint.unlocks.fabricator",
            FactType.CONTENT_RELATION,
            blueprint,
            fabricator,
            {"relation": "unlocks"},
        ),
        ImplementationFact(
            "blueprint.requires.registry",
            FactType.CONTENT_RELATION,
            blueprint,
            registry,
            {"relation": "requires"},
        ),
        ImplementationFact(
            "fabricator.displays.registry",
            FactType.CONTENT_RELATION,
            fabricator,
            registry,
            {"relation": "displays"},
        ),
        ImplementationFact(
            "registry.displays.fabricator",
            FactType.CONTENT_RELATION,
            registry,
            fabricator,
            {"relation": "displays"},
        ),
        ImplementationFact(
            "registry.upgrades.fabricator",
            FactType.CONTENT_RELATION,
            registry,
            fabricator,
            {"relation": "upgrades"},
        ),
    )

    normalized = _normalize_content_build_dependencies(modules, relations)
    by_id = {module.module_id: module for module in normalized}

    assert by_id[blueprint].depends_on == (registry,)
    assert by_id[fabricator].depends_on == ()
    assert by_id[registry].depends_on == ()
    assert by_id[recipe].depends_on == (blueprint,)


def test_unbound_structured_content_assets_are_deferred_without_losing_visual_intent() -> None:
    from types import SimpleNamespace

    from minecraft_mod_ai.authored_production import (
        _defer_unbound_structured_content_assets,
    )
    from minecraft_mod_ai.complete_spec import AssetRequest, ProductionModule

    for module_kind, asset_kind, render_kind in (
        ("gui", "gui", "gui.sprite"),
        ("entity", "entity", "entity.fixed_uv"),
    ):
        module = ProductionModule("screen_or_entity", module_kind, {})
        visual_spec = {
            "role": "ship interface",
            "silhouette": "",
            "materials": [],
            "motifs": [],
            "palette": {"primary": "#112233"},
        }
        asset = AssetRequest(
            asset_id="texture_structured_subject",
            kind=asset_kind,
            visual_description="main_color: #112233",
            render_kind=render_kind,
            subject_id="screen_or_entity",
            owner_module_id="screen_or_entity",
            visual_spec=visual_spec,
        )
        version_context = SimpleNamespace(facts={})

        modules, assets = _defer_unbound_structured_content_assets(
            (module,),
            (asset,),
            version_context,
        )

        assert assets == ()
        assert modules[0].config["visual_spec"] == visual_spec


def test_partial_structured_host_binding_is_not_silently_deferred() -> None:
    from types import SimpleNamespace

    from minecraft_mod_ai.authored_production import (
        _defer_unbound_structured_content_assets,
    )
    from minecraft_mod_ai.complete_spec import AssetRequest, ProductionModule

    module = ProductionModule("ship_ui", "gui", {})
    asset = AssetRequest(
        asset_id="texture_gui_ship_ui",
        kind="gui",
        visual_description="main_color: #445566",
        render_kind="gui.sprite",
        subject_id="ship_ui",
        owner_module_id="ship_ui",
        visual_spec={"role": "ship ui", "palette": {"primary": "#445566"}},
    )
    version_context = SimpleNamespace(
        facts={
            "resource_asset_bindings": {
                # Deliberately incomplete. Presence means the normal strict
                # resource validator must diagnose it rather than migration
                # silently dropping the request.
                "ship_ui": {"render_kind": "gui.sprite"},
            }
        }
    )

    modules, assets = _defer_unbound_structured_content_assets(
        (module,),
        (asset,),
        version_context,
    )

    assert modules == (module,)
    assert assets == (asset,)


def test_saved_long_content_asset_id_migrates_without_replan() -> None:
    from minecraft_mod_ai.authored_production import (
        _canonicalize_saved_content_asset_id,
    )
    from minecraft_mod_ai.spec_identity import SPEC_ID_RE

    raw = "texture_item_space_mode_ship_module_blueprint_registry_item_template"
    canonical = _canonicalize_saved_content_asset_id(raw, 0)

    assert canonical == (
        "texture_item_space_mode_ship_module_blueprint_regis_b180319a9b72"
    )
    assert SPEC_ID_RE.fullmatch(canonical)


def test_python_generator_job_persists_schema_valid_candidate_inputs() -> None:
    from types import SimpleNamespace

    from jsonschema import Draft202012Validator

    from minecraft_mod_ai.artifact_expansion import expand_facts_to_jobs
    from minecraft_mod_ai.canonical_schema_compiler import compile_leaf_schemas
    from minecraft_mod_ai.implementation_fact import ImplementationFact
    from minecraft_mod_ai.integrity_dispatcher import _canonical_inputs_for_job
    from minecraft_mod_ai.populate_version_artifact_rules import build_version_facts
    from minecraft_mod_ai.prompt_fact_types import FactType

    leaf = "minecraft/screen/registration"
    host_facts = build_version_facts("26.2", base_facts={})
    resolved = SimpleNamespace(
        minecraft="26.2",
        context_id="candidate-26.2",
        leaf_bindings=host_facts["leaf_bindings"],
    )
    fact = ImplementationFact(
        fact_id="market_ui.exists",
        fact_type=FactType.GUI_EXISTS,
        subject="market_ui",
        display_name="Market UI",
        source_clause="Render a server-authoritative market screen.",
    )

    [job] = expand_facts_to_jobs(
        (fact,),
        mod_id="space_mod",
        package_name="example.space",
        minecraft_version="26.2",
        version_context=resolved,
    )
    inputs = _canonical_inputs_for_job(job, {})

    assert inputs == job.deterministic_inputs["_canonical_inputs"]
    input_schema, _ = compile_leaf_schemas(leaf)
    Draft202012Validator(input_schema).validate(inputs)
    spec = inputs["screen_registration_input"]
    assert spec["context_id"] == "candidate-26.2"
    assert spec["side"] == "CLIENT"
    assert spec["target_path"].startswith("src/client/java/example/space/client/generated/")
    assert spec["slots"][0]["name"] == "artifact_source"

    from minecraft_mod_ai.model_output_atomicity_contract import (
        assert_strict_atomicity_bounds,
    )

    slot_schema = spec["slots"][0]["schema"]
    Draft202012Validator.check_schema(slot_schema)
    assert_strict_atomicity_bounds(
        slot_schema,
        surface="canonical GUI source candidate",
    )
