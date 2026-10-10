"""Authored plans must have the same fail-closed quality gate as v2 proposals."""
from __future__ import annotations

import pytest

from minecraft_mod_ai.authored_production import _bind_authored_quality_contract
from minecraft_mod_ai.complete_spec import ProductionModule
from minecraft_mod_ai.production_contract import (
    ProductionContractError,
    evaluate_quality_contract,
    quality_unresolved,
    validate_production_contract,
)
from minecraft_mod_ai.quality_evidence import compile_quality_evidence


def _fixture():
    design = {
        "authored_plan": {"requested_prompt": "Create a crystal item", "text": "Crystal item"},
        "_authored_execution_manifest": {"schema_version": "mmm/authored-execution-manifest-v2"},
    }
    module = ProductionModule("crystal_item", "item", {"feature": "crystal"})
    return design, (module,)


def test_authored_production_binds_real_v2_quality_contract_before_approval():
    design, modules = _fixture()
    bound, acceptance = _bind_authored_quality_contract(
        requested_prompt="Create a crystal item",
        design=design,
        modules=modules,
        assets=(),
        acceptance=("The crystal item is registered.",),
    )
    contract = bound["_production_contract"]
    assert bound is not design
    assert "_production_contract" not in design
    assert contract["contract_sha256"].startswith("sha256:")
    assert "The crystal item is registered." in acceptance
    validate_production_contract(contract, modules, acceptance, ())



def test_internal_authored_plan_states_are_not_public_requirements():
    design, modules = _fixture()
    design["authored_plan"].update({
        "structured_sections": {
            "assembly": {"status": "assembly_task_queued"},
        },
        "typed_plan_ir": {
            "scheduler": {"task_sha256": "sha256:" + "a" * 64},
        },
        "content_design": {
            "workflow": {"done_predicate": "all declared provides"},
        },
    })
    bound, acceptance = _bind_authored_quality_contract(
        requested_prompt="Create a crystal item",
        design=design,
        modules=modules,
        assets=(),
        acceptance=("The crystal item is registered.",),
    )
    contract = bound["_production_contract"]
    statements = [row["statement"] for row in contract["requirement_catalog"]]
    assert any(row["source_ref"] == "game_design:$.authored_plan.text"
               for row in contract["requirement_catalog"])
    assert "Crystal item" in " ".join(statements)
    assert not any("assembly_task_queued" in item for item in statements)
    assert not any("task_" in item.casefold() for item in acceptance)
    assert not any("done_predicate" in item for item in acceptance)
    validate_production_contract(contract, modules, acceptance, ())

    # Planner-only fields are excluded from *public* checks, but still
    # content-addressed; changing them must invalidate the quality binding.
    changed = {
        **bound,
        "authored_plan": {
            **bound["authored_plan"],
            "typed_plan_ir": {"scheduler": {"task_sha256": "sha256:" + "b" * 64}},
        },
    }
    with pytest.raises(ProductionContractError, match="game_design does not match"):
        compile_quality_evidence(
            contract,
            "sha256:" + "c" * 64,
            game_design=changed,
            source_validation=None,
            build_report=None,
            jar_validation=None,
        )


def test_authored_quality_without_evidence_stays_blocked_not_self_certified():
    design, modules = _fixture()
    bound, acceptance = _bind_authored_quality_contract(
        requested_prompt="Create a crystal item",
        design=design,
        modules=modules,
        assets=(),
        acceptance=("The crystal item is registered.",),
    )
    contract = bound["_production_contract"]
    proposal_hash = "sha256:" + "a" * 64
    receipts = compile_quality_evidence(
        contract,
        proposal_hash,
        game_design=bound,
        source_validation=None,
        build_report=None,
        jar_validation=None,
    )
    assert receipts == {}
    report = evaluate_quality_contract(contract, receipts, proposal_hash)
    assert report["overall_status"] == "MISSING"
    assert quality_unresolved(report)
    assert set(quality_unresolved(report)) >= {"correctness", "build"}


def test_authored_design_mutation_breaks_content_addressed_quality_binding():
    design, modules = _fixture()
    bound, acceptance = _bind_authored_quality_contract(
        requested_prompt="Create a crystal item",
        design=design,
        modules=modules,
        assets=(),
        acceptance=("The crystal item is registered.",),
    )
    assert acceptance
    mutated = {**bound, "authored_plan": {**bound["authored_plan"], "text": "Different item"}}
    with pytest.raises(ProductionContractError, match="game_design does not match"):
        compile_quality_evidence(
            bound["_production_contract"],
            "sha256:" + "b" * 64,
            game_design=mutated,
            source_validation=None,
            build_report=None,
            jar_validation=None,
        )


def test_authored_contract_cannot_be_silently_overwritten():
    design, modules = _fixture()
    with pytest.raises(ValueError, match="AUTHORED_QUALITY_CONTRACT_ALREADY_BOUND"):
        _bind_authored_quality_contract(
            requested_prompt="Create a crystal item",
            design={**design, "_production_contract": {}},
            modules=modules,
            assets=(),
            acceptance=("The crystal item is registered.",),
        )


def test_authored_behavioral_coverage_is_mandatory_for_v1_and_v2():
    from types import SimpleNamespace
    from minecraft_mod_ai.complete_orchestrator import _is_authored_coverage_proposal

    design, _ = _fixture()
    for version in ("mmm/complete-proposal-v1", "mmm/complete-proposal-v2"):
        proposal = SimpleNamespace(schema_version=version, game_design=design)
        assert _is_authored_coverage_proposal(proposal)
    assert not _is_authored_coverage_proposal(
        SimpleNamespace(game_design={"authored_plan": design["authored_plan"]})
    )


def test_old_saved_authored_plan_is_rejected_before_expensive_full_build():
    from types import SimpleNamespace
    from minecraft_mod_ai.complete_orchestrator import (
        CompleteProductionError,
        _require_authored_quality_before_generation,
    )

    design, modules = _fixture()
    old = SimpleNamespace(game_design=design)
    with pytest.raises(CompleteProductionError, match="AUTHORED_QUALITY_CONTRACT_REQUIRED_BEFORE_GENERATION"):
        _require_authored_quality_before_generation(old, source_only=False)
    # Source-only remains useful for diagnosing historical plans; it does not
    # turn a legacy plan into a verified releasable artifact.
    _require_authored_quality_before_generation(old, source_only=True)

    bound, _ = _bind_authored_quality_contract(
        requested_prompt="Create a crystal item",
        design=design,
        modules=modules,
        assets=(),
        acceptance=("The crystal item is registered.",),
    )
    _require_authored_quality_before_generation(
        SimpleNamespace(game_design=bound), source_only=False
    )
