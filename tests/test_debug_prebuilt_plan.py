"""Explicit prebuilt Debug exercises a fixed multi-artifact plan without model planning."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from minecraft_mod_ai.authored_plan import AuthoredPlan
from minecraft_mod_ai.authored_structured_design import normalize_structured_sections
from minecraft_mod_ai.authored_content_contract import content_owned_refs
from minecraft_mod_ai.colab_run_modes import FULL_MODE, run_plan_dialog
from minecraft_mod_ai.debug_prebuilt_plan import (
    build_prebuilt_debug_plan,
    write_prebuilt_debug_plan,
)
from minecraft_mod_ai.implementation_fact import ImplementationFact
from minecraft_mod_ai.resource_fact_inputs import resource_inputs
from minecraft_mod_ai.typed_host_capabilities import typed_host_capability_contracts
from minecraft_mod_ai.typed_plan_ir import validate_typed_plan_ir
from minecraft_mod_ai.typed_plan_support import assert_typed_plan_host_support


class _NoPlannerSession:
    def __init__(self) -> None:
        self.calls = []

    def plan(self, prompt):
        raise AssertionError(f"default Debug called the model planner: {prompt}")

    def save_plan(self, path):
        raise AssertionError("Default Debug should reuse prebuilt saved plan")

    def load_plan(self, path):
        self.calls.append(("load", str(path)))
        plan = AuthoredPlan.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        return SimpleNamespace(message=plan.text, complete_proposal=plan)


def test_explicit_debug_prebuilt_never_calls_model(tmp_path: Path) -> None:
    session = _NoPlannerSession()
    path = tmp_path / "proposal.json"
    result = run_plan_dialog(
        session=session,
        run_mode=FULL_MODE,
        prompt="an unrelated request must not be substituted into a canned plan",
        plan_path=path,
        debug_mode=True,
        debug_strategy="prebuilt",
        minecraft_version="1.21.8",
        loader="fabric",
        print_fn=lambda *args, **kwargs: None,
    )
    assert result.approved
    assert result.plan_path == path
    assert session.calls == [("load", str(path))]
    plan = result.reply.complete_proposal
    assert isinstance(plan, AuthoredPlan)
    assert "an unrelated request" not in plan.requested_prompt
    assert len(plan.content_design["modules"]) >= 3
    assert {row["kind"] for row in plan.content_design["modules"]} >= {"item", "block", "recipe"}
    assert not plan.content_design["_debug_reference_selection"]["donor_claims"]


def test_prebuilt_roundtrip_matches_production_plan_abi(tmp_path: Path) -> None:
    plan = build_prebuilt_debug_plan()
    filename = write_prebuilt_debug_plan(tmp_path / "prebuilt.json")
    restored = AuthoredPlan.from_dict(json.loads(filename.read_text(encoding="utf-8")))
    assert restored.calculate_hash() == plan.calculate_hash()
    assert normalize_structured_sections(restored.structured_sections) == restored.structured_sections
    assert validate_typed_plan_ir(
        restored.typed_plan_ir, capabilities=typed_host_capability_contracts()
    ) == restored.typed_plan_ir
    assert_typed_plan_host_support(
        restored.structured_sections,
        restored.typed_plan_ir,
        externally_covered_refs=content_owned_refs(
            restored.structured_sections, restored.content_design,
        ),
    )
    facts = [
        ImplementationFact.from_dict(raw)
        for raw in restored.content_design["_implementation_facts"]
    ]
    recipe = next(fact for fact in facts if fact.subject == "crystal_block_recipe")
    assert resource_inputs(recipe, "mmm_debug_crystal")[0] == "fabric/recipe/shaped"
    assert len(facts) == 3
    selected = restored.content_design["_debug_reference_selection"]
    assert selected["search_terms"]
    assert selected["verification_stage"] == "production_after_target_binding"
    assert selected["donor_claims"] == []


def test_prebuilt_reference_selection_reaches_proof_only_repository_cards(monkeypatch) -> None:
    from minecraft_mod_ai.authored_reuse_bridge import resolve_authored_source_reuse
    from minecraft_mod_ai.grounded_source_reuse import _grounded_repository_cards
    import minecraft_mod_ai.grounded_source_reuse as reuse

    plan = build_prebuilt_debug_plan()
    selections = plan.content_design["_debug_reference_selection"]["reference_candidates"]
    repositories = tuple(row["repository"] for row in selections)
    assert repositories == ("FabricMC/fabric-example-mod",)
    received = {}

    def fake_host_proof(design):
        received.update(design)
        cards = _grounded_repository_cards(design)
        selected = next(
            card for card in cards
            if card["repository"] == "FabricMC/fabric-example-mod"
        )
        assert selected["explicit_reference"] is True
        assert selected["reference_only"] is True
        assert selected["source_reuse_authority"] == "verification_required"
        return {
            "schema_version": "mmm/grounded-repository-reuse-plan-v2",
            "capabilities": [],
        }

    monkeypatch.setattr(reuse, "build_repository_reuse_plan", fake_host_proof)
    result = resolve_authored_source_reuse(
        plan.requested_prompt,
        plan.structured_sections,
        minecraft_version="1.21.8",
        loader="fabric",
        reference_repositories=repositories,
    )
    assert received["_preselected_reference_repositories"] == list(repositories)
    assert result["origin"] == "default_authored_production"
    assert result["bound_target"] == {
        "minecraft_version": "1.21.8", "loader": "fabric",
    }


def test_prebuilt_fact_graph_lowers_on_real_fabric_target_without_model() -> None:
    """Catch errors in saved-plan module/fact routing before expensive model use."""
    from minecraft_mod_ai.authored_production import _compile_content_artifact_graph
    from minecraft_mod_ai.platform_catalog import adapter_for_target

    plan = build_prebuilt_debug_plan()
    for version in ("1.21.8", "26.2"):
        adapter = adapter_for_target(version, "fabric")
        modules, assets, jobs = _compile_content_artifact_graph(
            plan,
            adapter=adapter,
            mod_id="mmm_debug_crystal",
            package_name="ai.minecraft.generated.mmm_debug_crystal",
        )
        assert not assets
        assert {module.module_id for module in modules} == {
            "crystal_fragment", "crystal_block", "crystal_block_recipe",
        }
        assert jobs
        assert {job.owner_module for job in jobs} == {
            "crystal_fragment", "crystal_block", "crystal_block_recipe",
        }
        recipe = next(job for job in jobs if job.owner_module == "crystal_block_recipe")
        block_registration = next(
            job for job in jobs
            if job.template_id == "fabric/block/register_basic"
            and job.owner_module == "crystal_block"
        )
        assert block_registration.deterministic_inputs["authored_block_strength"] == [2.0, 3.0]
        client_model = next(
            job for job in jobs
            if job.template_id == "minecraft/resource/item/client_block_item"
            and job.owner_module == "crystal_block"
        )
        assert client_model.requires == ("crystal_block.registry_id",)
        assert "crystal_block.block_registry_id" in block_registration.produces
        assert "crystal_block.registry_id" in block_registration.produces
        assert "crystal_block.registry_id" in recipe.requires
        assert "crystal_block.block_registry_id" not in recipe.requires
        assert {
            "name": "crystal_block.registry_id",
            "kind": "REGISTRY_ID",
            "target_type": "Item",
        } in recipe.required_ports



def test_debug_plan_path_does_not_overwrite_user_proposal(tmp_path: Path) -> None:
    from minecraft_mod_ai.colab_run_modes import resolve_plan_path

    regular = resolve_plan_path(run_mode=FULL_MODE, output_root=tmp_path)
    regular.parent.mkdir(parents=True, exist_ok=True)
    regular.write_text('{"kept":"user plan"}', encoding="utf-8")
    for strategy in ("prebuilt", "host_smoke", "model_path"):
        separate = resolve_plan_path(
            run_mode=FULL_MODE,
            output_root=tmp_path,
            configured_path=str(regular),
            debug_strategy=strategy,
        )
        assert separate != regular
        assert separate.parent == tmp_path / "debug"
        assert separate.name == f"{strategy}-proposal.json"
    debug_target = resolve_plan_path(
        run_mode=FULL_MODE, output_root=tmp_path, debug_strategy="prebuilt",
    )
    write_prebuilt_debug_plan(debug_target)
    assert regular.read_text(encoding="utf-8") == '{"kept":"user plan"}'
