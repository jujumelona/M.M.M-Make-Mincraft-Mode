from __future__ import annotations

from collections import Counter
from pathlib import Path

from minecraft_mod_ai.artifact_validators.reference import (
    validate_block_vertical_slice,
    validate_item_vertical_slice,
)
from minecraft_mod_ai.complete_orchestrator import (
    CompleteExecutionOptions,
    CompleteProductionOrchestrator,
)
from minecraft_mod_ai.complete_planner import _lower_implementation_facts_and_jobs
from minecraft_mod_ai.complete_spec import (
    AssetRequest,
    ProductionModule,
    complete_proposal_from_parts,
)
from minecraft_mod_ai.pipeline import MinecraftModPipeline
from minecraft_mod_ai.planner import HeuristicPlanner
from minecraft_mod_ai.production_contract import compile_production_contract
from minecraft_mod_ai.resource_contracts import derive_module_asset_specs


def _module_assets(modules: tuple[ProductionModule, ...]) -> tuple[AssetRequest, ...]:
    """Finalize deterministic module assets before the production contract is frozen."""
    return tuple(AssetRequest(**row) for row in derive_module_asset_specs(modules))


def test_planner_lowers_implementation_facts_and_jobs():
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan("Add raw lunite")
    modules = (
        ProductionModule(
            "raw_lunite",
            "item",
            {"name": "Raw Lunite", "stack_limit": 16, "color": "#808080"},
        ),
    )
    facts, jobs = _lower_implementation_facts_and_jobs(modules, base.spec)

    assert len(facts) == 2
    assert facts[0]["subject"] == "raw_lunite"
    assert facts[0]["fact_type"] == "ITEM_EXISTS"
    assert facts[0]["display_name"] == "Raw Lunite"
    assert facts[1]["fact_type"] == "ITEM_STACK_LIMIT"
    assert facts[1]["value"] == 16

    # Template names are version-owned implementation details. The stable lowering
    # contract is the canonical leaf decomposition and its exact multiplicity.
    assert len(jobs) == 7
    leaf_counts = Counter(job["canonical_leaf"] for job in jobs)
    assert leaf_counts == Counter({
        "minecraft/item/registry": 2,
        "minecraft/item/properties": 1,
        "minecraft/item/model": 2,
        "minecraft/item/language": 1,
        "minecraft/item/integration": 1,
    })
    assert all(job["context_id"] == base.spec.platform.version_context.context_id for job in jobs)
    assert all(job["implementation_id"] for job in jobs)


def test_orchestrator_executes_artifact_jobs_and_materializes_to_disk(tmp_path: Path):
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan("Add raw lunite item")
    mod_id = base.spec.mod_id
    pkg = base.spec.package_name

    modules = (
        ProductionModule(
            "raw_lunite",
            "item",
            {"name": "Raw Lunite", "stack_limit": 16, "color": "#808080"},
        ),
    )

    facts, jobs = _lower_implementation_facts_and_jobs(modules, base.spec)
    assets = _module_assets(modules)
    game_design = {
        "title": "Artifact Job Execution Mod",
        "_implementation_facts": facts,
        "_artifact_jobs": jobs,
    }
    compiled = compile_production_contract(
        requested_prompt="Add raw lunite item",
        game_design=game_design,
        modules=modules,
        assets=assets,
        acceptance_tests=("raw lunite exists",),
    )
    proposal = complete_proposal_from_parts(
        requested_prompt="Add raw lunite item",
        base_proposal=base,
        game_design={**game_design, "_production_contract": compiled.contract},
        modules=modules,
        assets=assets,
        acceptance_tests=compiled.acceptance_tests,
    )

    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path / "out")
    result = orchestrator.execute(
        proposal,
        approval_hash=proposal.calculate_hash(),
        run_name="artifact-item-run",
        options=CompleteExecutionOptions(
            source_only=True,
            run_jdt=False,
            run_blockbench=False,
            run_runtime=False,
        ),
    )

    assert result.status == "SOURCE_READY"
    project_root = Path(result.project_root)

    # Verify that the vertical slice files were generated and pass reference validation!
    val_result = validate_item_vertical_slice(
        project_root,
        mod_id=mod_id,
        item_name="raw_lunite",
        package_name=pkg,
        main_class=getattr(base.spec, "main_class", "") or "",
        expected_stack_limit=16,
    )
    assert val_result["status"] == "PASS"

    # Verify that receipt records artifact-graph execution
    assert any(
        r.get("schema_version") == "mmm/artifact-graph-execution-receipt-v1"
        for r in result.module_receipts
    )


def test_orchestrator_executes_artifact_jobs_for_blocks_and_materializes_to_disk(tmp_path: Path):
    base = MinecraftModPipeline(planner=HeuristicPlanner()).plan("Add lunite ore block")
    mod_id = base.spec.mod_id
    pkg = base.spec.package_name

    modules = (
        ProductionModule(
            "raw_lunite",
            "item",
            {"name": "Raw Lunite", "color": "#808080"},
        ),
        ProductionModule(
            "lunite_ore",
            "block",
            {"name": "Lunite Ore", "drop": "raw_lunite", "color": "#808080"},
        ),
    )

    facts, jobs = _lower_implementation_facts_and_jobs(modules, base.spec)
    assets = _module_assets(modules)
    game_design = {
        "title": "Artifact Block Execution Mod",
        "_implementation_facts": facts,
        "_artifact_jobs": jobs,
    }
    compiled = compile_production_contract(
        requested_prompt="Add lunite ore block",
        game_design=game_design,
        modules=modules,
        assets=assets,
        acceptance_tests=("lunite ore exists",),
    )
    proposal = complete_proposal_from_parts(
        requested_prompt="Add lunite ore block",
        base_proposal=base,
        game_design={**game_design, "_production_contract": compiled.contract},
        modules=modules,
        assets=assets,
        acceptance_tests=compiled.acceptance_tests,
    )

    orchestrator = CompleteProductionOrchestrator(workspace_root=tmp_path / "out")
    result = orchestrator.execute(
        proposal,
        approval_hash=proposal.calculate_hash(),
        run_name="artifact-block-run",
        options=CompleteExecutionOptions(
            source_only=True,
            run_jdt=False,
            run_blockbench=False,
            run_runtime=False,
        ),
    )

    assert result.status == "SOURCE_READY"
    project_root = Path(result.project_root)

    # Verify that the item slice was generated and validated
    val_item = validate_item_vertical_slice(
        project_root,
        mod_id=mod_id,
        item_name="raw_lunite",
        package_name=pkg,
        main_class=getattr(base.spec, "main_class", "") or "",
    )
    assert val_item["status"] == "PASS"

    # Verify that the block slice was generated and validated
    val_block = validate_block_vertical_slice(
        project_root,
        mod_id=mod_id,
        block_name="lunite_ore",
        package_name=pkg,
        main_class=getattr(base.spec, "main_class", "") or "",
        expected_drop_item="raw_lunite",
    )
    assert val_block["status"] == "PASS"

    # Verify receipt records both modules
    receipt = next(
        r for r in result.module_receipts
        if r.get("schema_version") == "mmm/artifact-graph-execution-receipt-v1"
    )
    assert "raw_lunite" in receipt["module_ids"]
    assert "lunite_ore" in receipt["module_ids"]
