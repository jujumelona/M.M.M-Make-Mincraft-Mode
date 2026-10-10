"""The exact pre-authored Colab Debug text ends with a newline.

Before the fix, compile_production_contract built a derived public acceptance
with that trailing newline, then immediately rejected its own catalog because
the canonical acceptance validator trims statements before comparing them.
"""
from minecraft_mod_ai.acceptance_contracts import public_acceptance_values
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.debug_prebuilt_plan import (
    DEBUG_PREBUILT_PROMPT,
    DEBUG_PREBUILT_TEXT,
)
from minecraft_mod_ai.production_contract import (
    compile_production_contract,
    validate_production_contract,
)


def test_debug_authored_multiline_text_compiles_and_keeps_exact_source_binding():
    assert DEBUG_PREBUILT_TEXT.endswith("\n")
    modules = (ProductionModule(
        module_id="crystal_fragment", kind="item",
        config={"display_name": "Crystal Fragment", "stack_limit": 64},
    ),)
    design = {"authored_plan": {
        "text": DEBUG_PREBUILT_TEXT,
        "requested_prompt": DEBUG_PREBUILT_PROMPT,
    }}
    compiled = compile_production_contract(
        requested_prompt=DEBUG_PREBUILT_PROMPT,
        game_design=design,
        modules=modules,
        acceptance_tests=("Crystal fragment can be registered.",),
    )
    acceptance_catalog = compiled.contract["acceptance_catalog"]
    assert tuple(public_acceptance_values(acceptance_catalog)) == compiled.acceptance_tests
    assert all(value == value.strip() for value in compiled.acceptance_tests)
    assert any(
        req["source_ref"] == "game_design:$.authored_plan.text"
        and req["statement"] == DEBUG_PREBUILT_TEXT
        for req in compiled.contract["requirement_catalog"]
    )
    assert any(
        row["origin"] == "requirement"
        and row["statement"].endswith("Failures must remain visible as failures.")
        for row in acceptance_catalog
    )
    validate_production_contract(
        compiled.contract, modules, compiled.acceptance_tests,
    )


def test_public_acceptance_with_newline_is_canonicalized_without_discading_test():
    modules = (ProductionModule("example", "item", {"display_name": "Example"}),)
    result = compile_production_contract(
        requested_prompt="Create an example item.",
        game_design={"authored_plan": {
            "text": "# Behavior\nCreate the item.\n\n",
        }},
        modules=modules,
        acceptance_tests=("Item exists.\n",),
    )
    assert result.acceptance_tests[0] == "Item exists."
    assert len(result.acceptance_tests) >= 2
    assert all(row["statement"] == row["statement"].strip()
               for row in result.contract["acceptance_catalog"] if row["visibility"] == "public")



def test_real_prebuilt_crystal_plan_binds_same_public_tests_to_proposal():
    from minecraft_mod_ai.authored_production import _bind_authored_quality_contract
    from minecraft_mod_ai.debug_prebuilt_plan import build_prebuilt_debug_plan

    plan = build_prebuilt_debug_plan()
    modules = tuple(ProductionModule(
        module_id=entry["module_id"],
        kind=entry["kind"],
        config=entry["config"],
        depends_on=tuple(entry["depends_on"]),
    ) for entry in plan.content_design["modules"]) + (
        ProductionModule("authored_typed_plan", "typed_host", {
            "typed_plan_ir": plan.typed_plan_ir,
        }),
    )
    bound, acceptance = _bind_authored_quality_contract(
        requested_prompt=plan.requested_prompt,
        design={"authored_plan": plan.to_dict()},
        modules=modules,
        assets=(),
        acceptance=(
            "Implement the behaviors in the saved authored design and exercise them in Minecraft.",
            "Build the project and verify that the mod loads and runs without errors.",
        ),
        extra_acceptance=plan.content_design["acceptance_tests"],
    )
    assert acceptance == tuple(public_acceptance_values(
        bound["_production_contract"]["acceptance_catalog"]
    ))
    assert any("Crafting recipe produces crystal_block" in value for value in acceptance)
    validate_production_contract(bound["_production_contract"], modules, acceptance)
