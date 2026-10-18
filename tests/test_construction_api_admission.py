"""Only plausible real APIs may enter the small model's choice set."""
from types import SimpleNamespace

from minecraft_mod_ai.construction_api_admission import (
    api_semantic_admissible, root_action_stems,
)
from minecraft_mod_ai.construction_operation_binding import _usable_api_candidate
from minecraft_mod_ai.construction_retrieval import ConstructionEvidenceIndex


def row(owner, name, result="void"):
    return {"kind": "METHOD", "owner": owner, "name": name,
            "return_type": result, "parameters": [], "access": 9, "static": True}


def test_root_registration_rejects_unrelated_bytecode_even_when_type_fits():
    purpose = "Register an alien entity type in the Minecraft registry"
    assert "regis" in root_action_stems(purpose)
    assert api_semantic_admissible("call", purpose, row("net.minecraft.core.Registry", "register"))
    assert api_semantic_admissible("call", purpose, row("example.EntityRegistry", "registerEntity"))
    assert not api_semantic_admissible("call", purpose, row(
        "net.minecraft.world.level.levelgen.DensityFunctions", "argument1"))
    assert not api_semantic_admissible("call", purpose, row(
        "net.minecraft.world.entity.Entity", "registerDebugValues"))
    assert not api_semantic_admissible("call", purpose, row(
        "net.minecraft.world.entity.Entity", "registerDefaultAttributes"))


def test_typed_intermediate_values_are_not_blocked_by_parent_action():
    # For argument values, the parent's exact Java type contract has priority.
    purpose = "Provide a required value for registration"
    assert api_semantic_admissible(
        "call", purpose, row("example.Source", "getValue"),
        [{"mode": "expression", "expected_type": "example.Target"}])


def test_binder_never_offers_unrelated_registration_candidate():
    chosen = row("net.minecraft.world.level.levelgen.DensityFunctions", "argument1")
    state = SimpleNamespace(evidence_index=SimpleNamespace(
        accessible=lambda key, constructor=False: True,
        intent_compatible=lambda key: True,
    ))
    assert not _usable_api_candidate(
        "DensityFunctions#argument1", chosen, op="call", kinds=("METHOD",),
        current_member="example.Mod#register", rejected_symbols=set(),
        state=state, fits=lambda t: True, requirements=(),
        purpose="Register an alien entity")
    good = row("net.minecraft.core.Registry", "register")
    assert _usable_api_candidate(
        "Registry#register", good, op="call", kinds=("METHOD",),
        current_member="example.Mod#register", rejected_symbols=set(),
        state=state, fits=lambda t: True, requirements=(),
        purpose="Register an alien entity")


def test_retrieval_does_not_expand_global_inventory_for_no_matching_action():
    owner = "example.Mod"
    unrelated = "net.minecraft.world.level.levelgen.DensityFunctions"
    symbols = {
        owner: {"kind": "CLASS", "name": "Mod", "owner": owner,
                "access": 1, "provenance": "construction_plan"},
        unrelated: {"kind": "CLASS", "name": "DensityFunctions",
                    "owner": unrelated, "access": 1},
        unrelated + "#argument1": row(unrelated, "argument1"),
    }
    index = ConstructionEvidenceIndex(symbols, {}, SimpleNamespace(),
        target_path="src/main/java/example/Mod.java", owner=owner,
        authoring_contract={})
    index._eligible_owners = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("No global API sweep for a specific action"))
    result = index.page(index.prepare(
        "Register alien entities", kinds=("METHOD",)), 0)
    assert result["matches"] == []
    assert result["total_count"] == 0


def test_register_requires_matching_registry_domain_not_just_name():
    task = "Register a hostile alien entity type"
    assert api_semantic_admissible(
        "call", task, row("net.minecraft.core.Registry", "register"))
    assert api_semantic_admissible(
        "call", task, row("net.minecraft.world.entity.EntityType", "register"))
    for unrelated in (
        "net.minecraft.world.item.crafting.RecipeType",
        "net.minecraft.world.level.levelgen.structure.templatesystem.RuleTestType",
        "net.minecraft.world.entity.ai.gossip.ReputationEventType",
        "net.fabricmc.fabric.api.client.rendering.v1.EntityRendererRegistry",
    ):
        assert not api_semantic_admissible("call", task, row(unrelated, "register"))

    # These all contain the word Entity, but register a different kind of
    # object or a client-only rendering hook. None creates an entity type.
    for unrelated in (
        "net.fabricmc.fabric.api.object.builder.v1.entity.FabricEntityDataRegistry",
        "net.fabricmc.fabric.api.registry.fluid.EntityFluidInteractionRegistry",
        "net.fabricmc.fabric.api.debug.v1.EntityDebugSubscriptionRegistry",
        "net.minecraft.client.renderer.entity.EntityRenderers",
    ):
        assert not api_semantic_admissible(
            "call", task, row(unrelated, "register")
        )

    assert api_semantic_admissible(
        "call", "Register a crafting recipe",
        row("net.minecraft.world.item.crafting.RecipeType", "register"))
